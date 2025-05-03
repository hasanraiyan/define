require('dotenv').config();
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const { OpenAI } = require('openai');
const { createLogger, format, transports } = require('winston');
const { v4: uuidv4 } = require('uuid');
const mongoose = require('mongoose'); // Mongoose for MongoDB
// console is a global object, no need to require it.

// --- Configuration & Setup ---

const PORT = process.env.PORT || 3001;
const MAX_RETRIES = parseInt(process.env.MAX_RETRIES || '10', 10);
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_BASE_URL = process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1';
const OPENAI_MODEL = process.env.OPENAI_MODEL || 'gpt-4o-mini';

const RATE_LIMIT_MAX_REQUESTS = parseInt(process.env.RATE_LIMIT_MAX_REQUESTS || '100', 10);
const RATE_LIMIT_WINDOW_MINUTES = parseInt(process.env.RATE_LIMIT_WINDOW_MINUTES || '15', 10);

const API_VERSION = "1.9.0"; // Version bump for lang parameter
const LOG_LEVEL = process.env.LOG_LEVEL || (process.env.NODE_ENV === 'production' ? 'info' : 'debug');

const DECAY_FACTOR = 0.9;

// MongoDB Configuration
const MONGODB_URI = process.env.MONGODB_URI;

// Exit immediately if API key is missing
if (!OPENAI_API_KEY) {
    console.error("FATAL ERROR: OPENAI_API_KEY environment variable is not set.");
    process.exit(1);
}

// --- Logger Setup ---
const logger = createLogger({
    level: LOG_LEVEL,
    format: format.combine(
        format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
        format.errors({ stack: true }),
        format.splat(),
        format.json()
    ),
    defaultMeta: { service: 'exact-word-definer-api' },
    transports: [
        new transports.File({
            filename: 'logs/error.log',
            level: 'error',
            maxsize: 5242880, // 5MB
            maxFiles: 5,
            tailable: true
        }),
        new transports.File({
            filename: 'logs/combined.log',
            maxsize: 10485760, // 10MB
            maxFiles: 10,
            tailable: true
        })
    ],
    exitOnError: false,
});

if (process.env.NODE_ENV !== 'production') {
    logger.add(new transports.Console({
        format: format.combine(
            format.colorize(),
            format.printf(({ timestamp, level, message, reqId, stack, ...meta }) => {
                const reqIdString = reqId ? `[${reqId}] ` : '';
                 // Filter out stack trace unless level is error/fatal
                const metaString = Object.keys(meta).length ? JSON.stringify(meta, (k, v) => k === 'stack' && level !== 'error' && level !== 'fatal' ? undefined : v, 2) : '';
                const stackString = stack && (level === 'error' || level === 'fatal') ? `\n${stack}` : '';
                return `${timestamp} [${level}] ${reqIdString}${message}${metaString}${stackString}`;
            })
        ),
        level: LOG_LEVEL
    }));
} else {
    logger.add(new transports.Console({
        format: format.json(),
        level: LOG_LEVEL
    }));
}

const app = express();

// --- Middleware Setup ---

app.use((req, res, next) => {
    req.id = req.headers['x-request-id'] || uuidv4();
    res.setHeader('X-Request-Id', req.id);
    req.logger = logger.child({ reqId: req.id });
    req.logger.info(`Incoming request: ${req.method} ${req.originalUrl}`, {
        httpMethod: req.method,
        url: req.originalUrl,
        ip: req.ip,
    });
    next();
});

app.use(helmet());
app.use(cors());
app.use(express.json());
// app.set('trust proxy', 1); // Uncomment if behind a proxy

const limiter = rateLimit({
    windowMs: RATE_LIMIT_WINDOW_MINUTES * 60 * 1000,
    max: RATE_LIMIT_MAX_REQUESTS,
    standardHeaders: 'draft-7',
    legacyHeaders: false,
    keyGenerator: (req) => {
        const word = req.params?.word || req.body?.word || req.query?.word || 'general';
        // Include lang in the rate limit key if desired, although IP+word is usually sufficient
        // const lang = (req.params?.lang || req.body?.lang || req.query?.lang || 'input_language').trim().toLowerCase();
        return `${req.ip || 'unknown'}|${word}`; // Optional: add `|${lang}` here
    },
    handler: (req, res) => {
        req.logger.warn('Rate limit exceeded', { ip: req.ip });
        res.status(429).json({
            error: `Too many requests for this word/IP combination, please try again later. Limit is ${RATE_LIMIT_MAX_REQUESTS} requests per ${RATE_LIMIT_WINDOW_MINUTES} minutes.`,
            requestId: req.id,
            retryAfterSeconds: Math.ceil(RATE_LIMIT_WINDOW_MINUTES * 60 / RATE_LIMIT_MAX_REQUESTS)
        });
    }
});


// --- Metrics Store ---
const metrics = {
    totalRequests: 0,
    requestsSucceededExact: 0, // Requests where exact word count was achieved (from API)
    requestsSucceededBestEffort: 0, // Requests where best effort result was returned (from API)
    requestsFailedInputValidation: 0,
    requestsFailedApiErrors: 0,
    requestsFailedServerErrors: 0,
    apiCallErrors: 0, // Total count of *individual* API call errors across all attempts
    cacheHits: 0, // New metric for cache hits (served directly from DB)
    cacheMisses: 0, // New metric for cache misses (went to API)
    cacheWrites: 0, // New metric for successful cache writes (any attempt)
    totalWordsRequested: 0, // Sum of requested lengths for requests handled by API
    totalWordsGenerated: 0, // Sum of *actual* lengths generated by API for successful (exact/best effort) requests
    totalWordsServedFromCache: 0, // Sum of actual lengths served from cache
    requestsByTone: {},
    requestsByContext: {},
    requestsByLang: {}, // New metric for language breakdown
    lastUpdated: null,
    startTime: new Date().toISOString(),
    dbConnectionStatus: 'disconnected'
};

// --- MongoDB Initialization and Schema ---
let isDbConnected = false;

const definitionSchema = new mongoose.Schema({
    word: { type: String, required: true, lowercase: true, trim: true },
    length: { type: Number, required: true, min: 1 }, // Stores the ACTUAL word count generated
    tone: { type: String, required: true, lowercase: true, default: 'neutral' },
    context: { type: String, lowercase: true, default: null }, // Store null for no context
    lang: { type: String, required: true, lowercase: true, trim: true, default: 'input_language' }, // New lang field
    definition: { type: String, required: true },
    createdAt: { type: Date, default: Date.now },
});

// Compound unique index now includes 'lang'
definitionSchema.index({ word: 1, length: 1, tone: 1, context: 1, lang: 1 }, { unique: true });

const Definition = mongoose.model('Definition', definitionSchema);

async function connectDB() {
    if (!MONGODB_URI) {
        logger.warn("MONGODB_URI not set. Running without database caching.");
        metrics.dbConnectionStatus = 'disabled';
        return;
    }

    try {
        await mongoose.connect(MONGODB_URI);
        isDbConnected = true;
        metrics.dbConnectionStatus = 'connected';
        logger.info('MongoDB connected successfully.');

        mongoose.connection.on('error', (err) => {
            logger.error('MongoDB connection error:', { error: err.message, stack: err.stack });
            isDbConnected = false;
            metrics.dbConnectionStatus = 'error';
        });
        mongoose.connection.on('disconnected', () => {
            logger.warn('MongoDB disconnected.');
            isDbConnected = false;
            metrics.dbConnectionStatus = 'disconnected';
        });
        mongoose.connection.on('connected', () => {
             if (!isDbConnected) { // Log re-connection
                 logger.info('MongoDB reconnected.');
                 isDbConnected = true;
                 metrics.dbConnectionStatus = 'connected';
             }
        });

    } catch (error) {
        logger.error('Failed to connect to MongoDB:', { error: error.message, stack: error.stack });
        isDbConnected = false;
        metrics.dbConnectionStatus = 'failed';
        // Allow the server to start even if DB connection fails
    }
}

// Initialize the database connection
connectDB();


// --- OpenAI Client Initialization ---
const openai = new OpenAI({
    apiKey: OPENAI_API_KEY,
    baseURL: OPENAI_BASE_URL,
    timeout: 30000,
    maxRetries: 2
});

logger.info(`OpenAI Client Initialized`, { model: OPENAI_MODEL, baseURL: OPENAI_BASE_URL, clientTimeout: openai.timeout, clientMaxRetries: openai.maxRetries });


// --- Helper Functions ---

function countWords(text) {
    if (!text || typeof text !== 'string') {
        return 0;
    }
    const cleanedText = text.trim().replace(/\s+/g, ' ').replace(/^(\*|-|\d+\.)\s+/, '');
    const words = cleanedText.split(/\s+/).filter(word => word.length > 0);
    return words.length;
}

// --- API Logic ---
async function handleDefineRequest(params, req, res) {
    const { logger: reqLogger, id: requestId } = req;

    metrics.totalRequests++;
    metrics.lastUpdated = new Date().toISOString();

    let attemptHistoryForThisRequest = [];
    let bestAttemptSoFar = null;
    let bestDifference = Infinity;
    let currentTemperature = 0.2;
    let cumulativeApiErrors = 0;
    let isCacheHit = false;

    try {
        // --- Parameter Parsing and Validation ---
        const { word } = params;
        const length = params.length ? parseInt(params.length, 10) : NaN;
        const tone = (params.tone || 'neutral').trim().toLowerCase();
        const validContexts = ['legal', 'educational', undefined];
        const contextRaw = params.context ? params.context.trim().toLowerCase() : undefined;
        const context = validContexts.includes(contextRaw) ? contextRaw : undefined; // Use undefined if invalid

        // New: Handle lang parameter
        // If lang is provided and not empty, use it. Otherwise, default to 'input_language'.
        const lang = (params.lang && params.lang.trim() !== '') ? params.lang.trim().toLowerCase() : 'input_language';


        if (!word || typeof word !== 'string' || word.trim().length === 0) {
             metrics.requestsFailedInputValidation++;
             reqLogger.warn('Invalid input: "word" parameter missing or empty');
             return res.status(400).json({
                 error: 'Invalid input. "word" parameter is required.',
                 requestId: requestId
            });
        }
        const term = word.trim().toLowerCase();

        if (isNaN(length) || length <= 0 || length > 250) {
             metrics.requestsFailedInputValidation++;
             reqLogger.warn('Invalid input: "length" parameter invalid', { lengthParam: params.length, parsedLength: length, params });
             return res.status(400).json({
                 error: 'Invalid input. "length" must be a positive integer (1 to 250).',
                 requestId: requestId
             });
        }
        const requestedLength = length;

        if (params.context && context === undefined) {
             reqLogger.warn('Invalid context parameter provided, ignoring.', { originalContext: params.context, validContexts });
        }
         // Log a warning if lang was provided but empty/whitespace
        if (params.lang && params.lang.trim() === '') {
             reqLogger.warn('Empty "lang" parameter provided, defaulting to input language.', { originalLang: params.lang });
        }


        // --- Metrics Update after successful input validation ---
        metrics.requestsByTone[tone] = (metrics.requestsByTone[tone] || 0) + 1;
        metrics.requestsByContext[context || 'none'] = (metrics.requestsByContext[context || 'none'] || 0) + 1;
        metrics.requestsByLang[lang] = (metrics.requestsByLang[lang] || 0) + 1; // Track requests by language

        reqLogger.info(`Processing request:`, { term, requestedLength, tone, context, lang });

        // --- Cache Lookup (Read) ---
        if (isDbConnected) {
            try {
                // Query the database - now includes 'lang' in the lookup
                const cachedDef = await Definition.findOne({
                    word: term,
                    length: requestedLength, // Look for an entry whose ACTUAL length matches the REQUESTED length
                    tone: tone,
                    context: context || null, // Match null if no context
                    lang: lang // Include language in cache key
                }).lean();

                if (cachedDef) {
                    reqLogger.info('Cache hit!', { term, requestedLength, actualLength: cachedDef.length, tone, context, lang });
                    metrics.cacheHits++;
                    metrics.totalWordsServedFromCache += cachedDef.length; // Track words served from cache
                    isCacheHit = true;

                    // Return the cached definition immediately
                    return res.json({
                        result: cachedDef.definition,
                        word: cachedDef.word,
                        requestedLength: requestedLength,
                        actualLength: cachedDef.length, // This will be same as requestedLength for a cache hit
                        status: "Cached result (Exact length achieved)",
                        attemptsMade: 0,
                        attemptsHistory: [],
                        summaryMetrics: {
                            totalTokensUsed: 0, avgTokensPerAttempt: 0, totalProcessingTime: "0ms", temperatureRange: { start: 0, end: 0 }
                         },
                        finalTemperature: 0,
                        requestId: requestId,
                        config: { tone, context, lang }, // Include lang in response config
                        cacheHit: true,
                        cachedAt: cachedDef.createdAt
                    });
                } else {
                    reqLogger.debug('Cache miss.', { term, requestedLength, tone, context, lang });
                    metrics.cacheMisses++;
                }
            } catch (dbError) {
                reqLogger.error('Database lookup error:', { error: dbError.message, stack: dbError.stack });
                metrics.dbConnectionStatus = 'error';
                // Continue without cache if lookup fails
            }
        } else {
            reqLogger.debug('Database not connected, skipping cache lookup.');
        }

        // --- Retry Logic Loop (Only runs on cache miss or if DB is down) ---
        metrics.totalWordsRequested += requestedLength; // Only count requested words if going to API

        let attemptsLeft = MAX_RETRIES;

        while (attemptsLeft > 0) {
            const attemptNumber = MAX_RETRIES - attemptsLeft + 1;
            reqLogger.debug(`Attempt ${attemptNumber}/${MAX_RETRIES}`, { term, requestedLength, currentTemperature: currentTemperature.toFixed(3) });

            // --- Dynamic Prompt Construction ---
            const previousAttemptsText = attemptHistoryForThisRequest
                .map((a, i) => `Attempt ${i + 1} Result (Actual words: ${a.actualWords}, Target: ${requestedLength}):\n"${a.output.replace(/"/g, '')}"`)
                .join('\n\n---\n');

            const styleInstructionString = tone && tone !== 'neutral' ? `Adopt a ${tone} tone.` : 'Maintain a neutral tone.';

            let contextInstructions = '';
            if (context === 'legal') {
                contextInstructions = 'Tailor the definition for a legal context. Focus on statutory definitions, legal precedents, and legal implications.';
            } else if (context === 'educational') {
                 contextInstructions = 'Tailor the definition for an educational context. Prioritize pedagogical clarity, learning objectives, and accessibility for students.';
            }

            // New: Dynamic Language Instruction
            const langInstructionString = lang === 'input_language' ?
                                          `Define the term in the language of the provided term ("${term}").` :
                                          `Define the term in ${lang}.`;


            const prompt = `Define "${term}" in EXACTLY ${requestedLength} words.

            ${langInstructionString}
            ${styleInstructionString}
            ${contextInstructions ? contextInstructions + '\n' : ''}

            REVISION INSTRUCTIONS based on previous attempts:
            - Analyze the "CURRENT ATTEMPTS" below.
            - If previous attempt was SHORT by 1-3 words: Add one precise adjective, clarifying phrase, or relevant detail to slightly increase length.
            - If previous attempt was LONG by 1-3 words: Remove redundant adverbs, parentheticals, or less critical details to slightly decrease length.
            - If the word count difference is larger, carefully restructure sentences or add/remove substantial points while maintaining accuracy.
            - NEVER sacrifice accuracy or clarity for word count or language adherence.

            EXAMPLE FOR 100 WORDS (Adjust structure approximately to match these proportions):
            [~20 words: concise context or introduction]. [~60 words: core definition with key details]. [~20 words: significance statement or application].

            CURRENT ATTEMPTS (Analyze this for revisions, but do not repeat or refer to it in your final response):
            ${previousAttemptsText || 'No previous attempts provided yet.'}

            RESPONSE MUST:
            - Be exactly ${requestedLength} words.
            - Consist ONLY of the definition text.
            - Use complete sentences.
            - Avoid quotation marks around the entire definition.
            - Exclude introductory phrases like "Definition:", "Here is a definition:", "The definition is:", etc.`;
            // --- End Dynamic Prompt Construction ---

            // --- Dynamic System Message Construction ---
            const systemMessageContent = `You are a highly precise language assistant trained to generate text definitions that adhere strictly to a given EXACT word count (${requestedLength} words).` +
                                         ` ${lang === 'input_language' ? `Define in the language of the input term.` : `Define in ${lang}.`}` + // Include language instruction
                                         ` ${styleInstructionString}` +
                                         `${contextInstructions ? ' ' + contextInstructions : ''}` +
                                         ` Analyze the user's prompt, paying close attention to revision instructions based on previous attempts. Your response must be the definition text only, exactly ${requestedLength} words long, and in the requested language.`;
            // --- End Dynamic System Message Construction ---


            try {
                const startTime = Date.now();
                const response = await openai.chat.completions.create({
                    model: OPENAI_MODEL,
                    messages: [
                        { role: 'system', content: systemMessageContent },
                        { role: 'user', content: prompt }
                    ],
                    temperature: currentTemperature,
                    max_tokens: 4096,
                    top_p: 1,
                    frequency_penalty: 0,
                    presence_penalty: 0,
                });
                const durationMs = Date.now() - startTime;

                const outputRaw = response.choices[0]?.message?.content || '';
                let output = outputRaw.trim();
                 if ((output.startsWith('"') && output.endsWith('"')) || (output.startsWith("'") && output.endsWith("'"))) {
                     output = output.substring(1, output.length - 1).trim();
                 }
                 output = output.replace(/^(\*|-|\d+\.)\s+/, '').trim();

                const actualWords = countWords(output);
                const usage = response.usage;

                reqLogger.debug(`LLM Response Received`, {
                    attempt: attemptNumber,
                    actualWords,
                    targetWords: requestedLength,
                    difference: actualWords - requestedLength,
                    durationMs,
                    modelUsed: response.model,
                    tokenUsage: usage,
                    outputPreview: `"${output.substring(0, 100)}${output.length > 100 ? '...' : ''}"`
                });

                if (actualWords === 0 && output.length > 0) {
                    reqLogger.warn(`Word count is 0 but output exists. Output: "${output}"`);
                }

                const attemptData = {
                    attempt: attemptNumber,
                    temperature: parseFloat(currentTemperature.toFixed(3)),
                    output: output,
                    actualWords: actualWords,
                    targetWords: requestedLength,
                    timestamp: new Date().toISOString(),
                    durationMs,
                    tokenUsage: usage,
                };
                attemptHistoryForThisRequest.push(attemptData);

                // --- Cache Write (Every Attempt with Output) ---
                // We save every generated definition that has words, alongside its *actual* word count, tone, context, and language.
                if (isDbConnected && actualWords > 0) {
                    try {
                        await Definition.create({
                            word: term,
                            length: actualWords, // IMPORTANT: Store the ACTUAL length generated in this attempt
                            tone: tone,
                            context: context || null,
                            lang: lang, // Store the language
                            definition: output,
                        });
                        reqLogger.debug('Cached definition from attempt.', { attempt: attemptNumber, term, length: actualWords, tone, context, lang });
                        metrics.cacheWrites++;
                    } catch (dbWriteError) {
                        if (dbWriteError.code === 11000) { // Duplicate key error
                            reqLogger.debug('Attempted to cache definition, but a duplicate already existed for this word, actual length, tone, context, lang.', { attempt: attemptNumber, term, length: actualWords, tone, context, lang });
                        } else {
                            reqLogger.error('Database write error (attempt cache):', { error: dbWriteError.message, stack: dbWriteError.stack });
                            metrics.dbConnectionStatus = 'error';
                        }
                    }
                } else if (!isDbConnected) {
                    reqLogger.debug('Database not connected, skipping cache write for attempt.');
                }
                // --- End Cache Write (Every Attempt) ---


                const difference = Math.abs(actualWords - requestedLength);
                 // Update best attempt if this one is closer AND has generated output words
                if (actualWords > 0 && difference < bestDifference) {
                    bestDifference = difference;
                    bestAttemptSoFar = attemptData;
                    reqLogger.debug(`New best attempt found`, { attempt: attemptNumber, difference, actualWords });
                }
                 // If no best attempt found yet, but this attempt has output, make it the best so far
                 else if (!bestAttemptSoFar && actualWords > 0) {
                     bestDifference = difference; // Calculate difference for this attempt
                     bestAttemptSoFar = attemptData;
                     reqLogger.debug(`First attempt with output found`, { attempt: attemptNumber, actualWords });
                 }


                // --- Success Condition ---
                if (actualWords === requestedLength) {
                    reqLogger.info(`Exact word count achieved`, { term, requestedLength, attemptsMade: attemptNumber });
                    metrics.requestsSucceededExact++;
                    metrics.totalWordsGenerated += actualWords; // Count generated words for API success

                    // Note: The caching for the exact match is already handled by the "Cache Write (Every Attempt)" block above.

                    res.json({
                        result: output,
                        word: term,
                        requestedLength: requestedLength,
                        actualLength: actualWords,
                        status: "Exact length achieved",
                        attemptsMade: attemptNumber,
                        attemptsHistory: attemptHistoryForThisRequest.map(a => ({
                            attempt: a.attempt,
                            actualWords: a.actualWords,
                            difference: a.actualWords - requestedLength,
                            temperature: a.temperature,
                            tokensUsed: a.tokenUsage,
                            durationMs: a.durationMs
                        })),
                         summaryMetrics: {
                             totalTokensUsed: attemptHistoryForThisRequest.reduce((sum, a) => sum + (a.tokenUsage?.total_tokens || 0), 0),
                             avgTokensPerAttempt: parseFloat((attemptHistoryForThisRequest.reduce((sum, a) => sum + (a.tokenUsage?.total_tokens || 0), 0) / attemptNumber).toFixed(2)),
                             totalProcessingTime: `${attemptHistoryForThisRequest.reduce((sum, a) => sum + a.durationMs, 0)}ms`,
                             temperatureRange: {
                                 start: parseFloat(attemptHistoryForThisRequest[0]?.temperature?.toFixed(3) || '0.2'),
                                 end: parseFloat(currentTemperature.toFixed(3))
                             }
                         },
                        finalTemperature: parseFloat(currentTemperature.toFixed(3)),
                        requestId: requestId,
                        config: { tone, context, lang }, // Include lang in response config
                        cacheHit: false
                    });
                    return; // Exit the function
                }

                // --- Temperature Adjustment ---
                if (attemptsLeft > 1) {
                     if (actualWords > 0) { // Only adjust temperature meaningfully if output had words
                        const error = actualWords - requestedLength;
                        let adjustment = Math.sign(error) * Math.min( Math.abs(error) * 0.05, 0.15 );
                        if (attemptHistoryForThisRequest.length > 1) {
                            const prevError = attemptHistoryForThisRequest[attemptHistoryForThisRequest.length - 2].actualWords - requestedLength;
                            adjustment += (prevError * 0.08) * DECAY_FACTOR;
                        }
                        currentTemperature = Math.min(0.7, Math.max(0.1, currentTemperature - adjustment));
                        currentTemperature += (Math.random() - 0.5) * 0.05;
                        currentTemperature = Math.min(0.7, Math.max(0.1, currentTemperature));
                        currentTemperature = parseFloat(currentTemperature.toFixed(2));
                        reqLogger.debug(`Temperature adjustment applied`, { attempt: attemptNumber, actualWords, requestedLength, error, adjustment: adjustment.toFixed(4), newTemp: currentTemperature.toFixed(3) });
                     } else {
                         // If 0 words, slightly randomize temperature to try a different path
                          currentTemperature = Math.min(0.7, Math.max(0.1, currentTemperature + (Math.random() * 0.1 - 0.02)));
                          currentTemperature = parseFloat(currentTemperature.toFixed(2));
                           reqLogger.debug(`Temperature randomized slightly due to 0 word output`, { attempt: attemptNumber, newTemp: currentTemperature.toFixed(3) });
                     }
                } else {
                    reqLogger.debug(`Last attempt (${attemptNumber}/${MAX_RETRIES}), no temperature adjustment.`, { currentTemperature: currentTemperature.toFixed(3) });
                }

            } catch (err) {
                cumulativeApiErrors++;
                metrics.apiCallErrors++;

                reqLogger.error(`OpenAI API error on attempt ${attemptNumber}`, {
                    error: { message: err.message, status: err.status, type: err.type, code: err.code },
                    attempt: attemptNumber, model: OPENAI_MODEL, currentTemperature: currentTemperature.toFixed(3)
                });

                // --- Backoff ---
                if (err.status === 429 || err.message?.toLowerCase().includes('rate limit')) {
                    reqLogger.warn(`API Rate limit hit (status ${err.status || 'N/A'}). Waiting...`);
                    if (OPENAI_BASE_URL.includes('pollinations')) reqLogger.warn('Pollinations rate limit likely hit.');
                    await new Promise(resolve => setTimeout(resolve, 2000 + Math.random() * 1500));
                } else if (err.status >= 500) {
                    reqLogger.warn(`API Server error (${err.status}). Waiting...`);
                    await new Promise(resolve => setTimeout(resolve, 2500 + Math.random() * 2000));
                } else if (err.code === 'ETIMEDEOUT' || err.code === 'ECONNABORTED') {
                    reqLogger.warn(`API Timeout/Network error (${err.code}). Waiting...`);
                    await new Promise(resolve => setTimeout(resolve, 1500 + Math.random() * 1000));
                 } else if (err.status === 401 || err.status === 403) {
                     reqLogger.fatal(`API Authentication/Permission Error (${err.status}). Aborting retries.`, { errorDetails: err.message });
                     attemptsLeft = 0;
                     metrics.requestsFailedApiErrors++;
                 } else {
                     reqLogger.error(`Unrecoverable API Client Error (${err.status || err.code}). Aborting retries.`, { errorDetails: err.message });
                     attemptsLeft = 0;
                      metrics.requestsFailedApiErrors++;
                 }
            } finally {
                attemptsLeft--;
            }
        } // End of while loop

        // --- Handle Failure Cases (After all retries exhausted) ---
        // If a best attempt was found (one that produced > 0 words)
        if (bestAttemptSoFar && bestAttemptSoFar.actualWords > 0) {
            reqLogger.warn(`Exact word count not achieved after ${MAX_RETRIES} attempts. Returning best effort result.`, {
                term, requestedLength, bestDifference, actualWords: bestAttemptSoFar.actualWords, lang
            });
            metrics.requestsSucceededBestEffort++;
            metrics.totalWordsGenerated += bestAttemptSoFar.actualWords; // Count generated words for API success

            // Note: The caching for this best effort result is already handled by the "Cache Write (Every Attempt)" block inside the loop.

            res.json({
                result: bestAttemptSoFar.output,
                word: term,
                requestedLength: requestedLength,
                actualLength: bestAttemptSoFar.actualWords,
                status: `Best effort result (Closest: ${bestAttemptSoFar.actualWords} words, Difference: ${bestDifference})`,
                attemptsMade: MAX_RETRIES,
                attemptsHistory: attemptHistoryForThisRequest.map(a => ({
                    attempt: a.attempt,
                    actualWords: a.actualWords,
                    targetDifference: a.actualWords - requestedLength,
                    temperature: a.temperature,
                    tokensUsed: a.tokenUsage,
                    durationMs: a.durationMs,
                })),
                summaryMetrics: {
                    totalTokensUsed: attemptHistoryForThisRequest.reduce((sum, a) => sum + (a.tokenUsage?.total_tokens || 0), 0),
                    avgTokensPerAttempt: parseFloat((attemptHistoryForThisRequest.reduce((sum, a) => sum + (a.tokenUsage?.total_tokens || 0), 0) / MAX_RETRIES).toFixed(2)),
                    totalProcessingTime: `${attemptHistoryForThisRequest.reduce((sum, a) => sum + a.durationMs, 0)}ms`,
                    temperatureRange: {
                         start: parseFloat(attemptHistoryForThisRequest[0]?.temperature?.toFixed(3) || '0.2'),
                         end: parseFloat(currentTemperature.toFixed(3))
                    }
                },
                finalTemperature: parseFloat(currentTemperature.toFixed(3)),
                requestId: requestId,
                config: { tone, context, lang }, // Include lang in response config
                cacheHit: false
            });
        } else {
            // No attempt produced valid output after all retries
            reqLogger.error(`All ${MAX_RETRIES} attempts failed or produced empty output for definition request.`, {
                term, requestedLength, cumulativeApiErrors, totalAttemptsMade: attemptHistoryForThisRequest.length, lang
            });
            metrics.requestsFailedApiErrors++;

            res.status(503).json({
                error: "Failed to generate definition after multiple attempts. The AI model may be unable to meet the exact requirements or there are upstream API issues. Please check input or try again later.",
                word: term,
                requestedLength: requestedLength,
                attemptsMade: attemptHistoryForThisRequest.length,
                apiErrorsDuringAttempts: cumulativeApiErrors,
                requestId: requestId,
                config: { tone, context, lang } // Include lang in response config
            });
        }

    } catch (error) {
        reqLogger.error('Critical Server Error in handleDefineRequest processing logic', {
            error: { message: error.message, stack: error.stack, name: error.name },
            requestParams: params,
            config: { tone, context, lang } // Include lang in error log config
        });
        metrics.requestsFailedServerErrors++;

        res.status(500).json({
            error: 'An internal server error occurred while processing the definition request.',
            requestId: requestId,
            timestamp: new Date().toISOString(),
            config: { tone, context, lang } // Include lang in response config
        });
    }
};


// --- API Endpoints ---
app.post('/api/define', limiter, (req, res) => {
    handleDefineRequest(req.body, req, res);
});

app.get('/api/define', limiter, (req, res) => {
    handleDefineRequest(req.query, req, res);
});

// --- Metrics Endpoint ---
app.get('/api/metrics', (req, res) => {
    const reqLogger = req.logger || logger.child({ reqId: req.id || 'metrics' });
    reqLogger.debug('Metrics endpoint requested');

    const now = new Date();
    const uptimeSeconds = process.uptime();
    // Total completed requests (API exact + API best effort + cache hits)
    const totalCompletedRequests = metrics.requestsSucceededExact + metrics.requestsSucceededBestEffort + metrics.cacheHits;
    const totalFailedRequests = metrics.requestsFailedInputValidation + metrics.requestsFailedApiErrors + metrics.requestsFailedServerErrors;
    const totalProcessedRequests = totalCompletedRequests + totalFailedRequests; // Total requests that reached handleDefineRequest

    // Calculate success rates based on total requests processed
    const successRateExactApi = totalProcessedRequests > 0 ? (metrics.requestsSucceededExact / totalProcessedRequests * 100) : 0; // Exact match from API only
    const successRateIncludingBestEffortAndCache = totalProcessedRequests > 0 ? (totalCompletedRequests / totalProcessedRequests * 100) : 0; // Any successful response (API or cache)

    // These averages/totals now ONLY count requests that went to the API and succeeded (not cache hits)
    const nonCachedSuccessfulRequests = metrics.requestsSucceededExact + metrics.requestsSucceededBestEffort;
    const avgWordsRequestedOnApiSuccess = parseFloat((metrics.totalWordsRequested / (nonCachedSuccessfulRequests || 1)).toFixed(1)); // Avoid div by zero
    const avgWordsGeneratedOnApiSuccess = parseFloat((metrics.totalWordsGenerated / (nonCachedSuccessfulRequests || 1)).toFixed(1)); // Avoid div by zero


    res.json({
        apiVersion: API_VERSION,
        serverStartTime: metrics.startTime,
        currentTime: now.toISOString(),
        uptime: `${uptimeSeconds.toFixed(1)} seconds`,
        memoryUsage: process.memoryUsage(),
        totalRequestsReceived: metrics.totalRequests,
        processedRequestsSummary: {
            total: totalProcessedRequests, // Total requests that reached handleDefineRequest
            succeededExactApi: metrics.requestsSucceededExact,
            succeededBestEffortApi: metrics.requestsSucceededBestEffort,
            cacheHits: metrics.cacheHits, // Requests successfully served from cache
            failedInputValidation: metrics.requestsFailedInputValidation,
            failedApiErrors: metrics.requestsFailedApiErrors, // Failures due to upstream API issues after retries
            failedServerErrors: metrics.requestsFailedServerErrors, // Failures due to internal logic errors
        },
        apiInteraction: {
            totalIndividualApiCallErrors: metrics.apiCallErrors, // Sum of errors across *all* LLM attempts
             cacheMetrics: { // Breakdown of cache interactions
                 cacheHits: metrics.cacheHits,
                 cacheMisses: metrics.cacheMisses,
                 cacheWrites: metrics.cacheWrites, // Total cache writes from any attempt
                 dbConnectionStatus: metrics.dbConnectionStatus,
             }
        },
        wordCounts: {
            averageWordsRequestedOnApiSuccess: avgWordsRequestedOnApiSuccess,
            averageWordsGeneratedOnApiSuccess: avgWordsGeneratedOnApiSuccess,
            totalWordsRequestedOnApiSuccess: metrics.totalWordsRequested,
            totalWordsGeneratedOnApiSuccess: metrics.totalWordsGenerated,
            totalWordsServedFromCache: metrics.totalWordsServedFromCache // Total words served from cache
        },
        successRates: {
            exactMatchApi: `${successRateExactApi.toFixed(1)}%`,
            includingBestEffortAndCache: `${successRateIncludingBestEffortAndCache.toFixed(1)}%`,
        },
        requestsBreakdown: {
            byTone: metrics.requestsByTone,
            byContext: metrics.requestsByContext,
            byLang: metrics.requestsByLang // Include language breakdown
        },
        lastMetricUpdate: metrics.lastUpdated,
        config: {
            model: OPENAI_MODEL,
            baseURL: OPENAI_BASE_URL,
            maxRetriesPerRequest: MAX_RETRIES,
            rateLimitMaxRequests: RATE_LIMIT_MAX_REQUESTS,
            rateLimitWindowMinutes: RATE_LIMIT_WINDOW_MINUTES,
            logLevel: LOG_LEVEL,
            database: MONGODB_URI ? 'MongoDB (Enabled)' : 'Disabled'
        }
    });
});


// --- Catch-all for 404 Not Found ---
app.use((req, res) => {
    const reqLogger = req.logger || logger.child({ reqId: req.id || 'N/A' });
    reqLogger.warn('Route not found', { url: req.originalUrl, method: req.method });
    res.status(404).json({
        error: 'Not Found',
        message: `The requested path ${req.originalUrl} does not exist on this server.`,
        requestId: req.id || 'N/A',
        timestamp: new Date().toISOString()
    });
});

// --- Final Error Handler Middleware ---
app.use((err, req, res, next) => {
    const errorLogger = req.logger || logger.child({ reqId: req.id || 'N/A' });
    const reqId = req.id || 'N/A';

    errorLogger.error('Unhandled error caught by final error handler', {
        reqId: reqId,
        error: {
            message: err.message,
            stack: err.stack,
            status: err.status,
            name: err.name
        },
        url: req.originalUrl,
        method: req.method,
        ip: req.ip
    });

    if (res.headersSent) {
        return next(err);
    }

    const statusCode = err.status || 500;
    const errorMessage = process.env.NODE_ENV === 'production' && statusCode >= 500
                         ? 'An unexpected server error occurred.'
                         : err.message;

    res.status(statusCode).json({
        error: statusCode === 404 ? 'Not Found' : (statusCode >= 500 ? 'Internal Server Error' : 'Client Error'),
        message: errorMessage,
        requestId: reqId,
        timestamp: new Date().toISOString()
    });
});

// --- Server Start ---
const server = app.listen(PORT, async () => {
    logger.info(`-----------------------------------------`);
    logger.info(` Exact Word Definer API Starting...`);
    logger.info(` Version: ${API_VERSION}`);
    logger.info(` Environment: ${process.env.NODE_ENV || 'development'}`);
    logger.info(` Log Level: ${LOG_LEVEL}`);
    logger.info(` Port: ${PORT}`);
    logger.info(` OpenAI Model: ${OPENAI_MODEL}`);
    logger.info(` OpenAI Base URL: ${OPENAI_BASE_URL}`);
    logger.info(` Max Retries (per request): ${MAX_RETRIES}`);
    logger.info(` Rate Limit: ${RATE_LIMIT_MAX_REQUESTS} req / ${RATE_LIMIT_WINDOW_MINUTES} min per word/IP`);
    logger.info(` Security Headers: Enabled`);
    logger.info(` CORS: Enabled`);
    logger.info(` Database: ${MONGODB_URI ? 'MongoDB (Attempting connection)' : 'Disabled (MONGODB_URI not set)'}`);

    // Warn if using default OpenAI base URL but model suggests Pollinations or vice versa
    if (OPENAI_BASE_URL.includes('api.openai.com') && (OPENAI_MODEL.includes('pollinations') || process.env.OPENAI_BASE_URL === undefined || process.env.OPENAI_BASE_URL.includes('pollinations.ai'))) {
         logger.warn(`Config Warning: Using OpenAI base URL '${OPENAI_BASE_URL}' but model '${OPENAI_MODEL}' or environment suggests Pollinations. Ensure OPENAI_BASE_URL is set correctly for your provider.`);
    } else if (OPENAI_BASE_URL.includes('pollinations.ai') && !OPENAI_MODEL.includes('pollinations') && process.env.OPENAI_MODEL === undefined) {
         logger.warn(`Config Warning: Using Pollinations base URL '${OPENAI_BASE_URL}' but model '${OPENAI_MODEL}' or environment suggests OpenAI. Ensure OPENAI_MODEL is set correctly for your provider.`);
    }
    logger.info(`-----------------------------------------`);

    metrics.startTime = new Date().toISOString();
});

// --- Process Error Handling ---
process.on('unhandledRejection', (reason, promise) => {
    logger.fatal('Unhandled Rejection at:', { promise, reason: reason?.message || reason, stack: reason?.stack });
    // Consider exiting for serious unhandled rejections
    // process.exit(1);
});

process.on('uncaughtException', (error) => {
    logger.fatal('Uncaught Exception:', { error: { message: error.message, stack: error.stack } });
    logger.on('finish', () => {
        process.exit(1);
    });
    setTimeout(() => {
      process.exit(1);
    }, 500);
});

process.on('SIGINT', () => gracefulShutdown('SIGINT'));
process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));

async function gracefulShutdown(signal) {
    logger.info(`Received ${signal}. Starting graceful shutdown.`);

    // Close HTTP server first
    server.close((err) => {
        if (err) {
            logger.error('Error during HTTP server close', { error: err.message, stack: err.stack });
        }
        logger.info('HTTP server closed.');

        // Then close database connection if it's open
        if (isDbConnected) {
            mongoose.connection.close(false) // false means do not force close connections
                .then(() => {
                    logger.info('MongoDB connection closed.');
                    isDbConnected = false;
                    metrics.dbConnectionStatus = 'disconnected';
                    logger.info('Graceful shutdown complete. Exiting.');
                    process.exit(0);
                })
                .catch((dbCloseErr) => {
                    logger.error('Error during MongoDB connection close', { error: dbCloseErr.message, stack: dbCloseErr.stack });
                    // Exit even if DB close fails
                    process.exit(1);
                });
        } else {
            logger.info('No active MongoDB connection to close.');
            logger.info('Graceful shutdown complete. Exiting.');
            process.exit(0);
        }
    });

    // Force shutdown after a timeout
    setTimeout(() => {
        logger.error('Graceful shutdown timed out. Forcing exit.');
        process.exit(1);
    }, 15000); // Increased timeout slightly
}