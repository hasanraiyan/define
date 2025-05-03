require('dotenv').config();
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const { OpenAI } = require('openai');
const { createLogger, format, transports } = require('winston'); // Winston for logging
const { v4: uuidv4 } = require('uuid'); // For request IDs
const { console } = require('inspector');

// --- Configuration & Setup ---

const PORT = process.env.PORT || 3001;
const MAX_RETRIES = parseInt(process.env.MAX_RETRIES || '10', 10);
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_BASE_URL = process.env.OPENAI_BASE_URL || 'https://text.pollinations.ai/openai?referrer=ExactWordDefinerBot';
const OPENAI_MODEL = process.env.OPENAI_MODEL || 'gpt-4o-mini';

const RATE_LIMIT_MAX_REQUESTS = parseInt(process.env.RATE_LIMIT_MAX_REQUESTS || '100', 10);
const RATE_LIMIT_WINDOW_MINUTES = parseInt(process.env.RATE_LIMIT_WINDOW_MINUTES || '15', 10);

const API_VERSION = "1.4.0"; // Updated version for logging changes
const LOG_LEVEL = process.env.LOG_LEVEL || (process.env.NODE_ENV === 'production' ? 'info' : 'debug');
const DECAY_FACTOR = 0.9; // Define decay factor for temperature momentum

if (!OPENAI_API_KEY) {
    console.error("FATAL ERROR: OPENAI_API_KEY environment variable is not set."); // Keep initial console for fatal startup errors
    process.exit(1);
}

// --- Logger Setup ---
const logger = createLogger({
    level: LOG_LEVEL,
    format: format.combine(
        format.timestamp(),
        format.errors({ stack: true }), // Log stack traces for errors
        format.splat(), // Allows for string interpolation like logger.info('User %s logged in', userId)
        format.json()   // Output logs as JSON
    ),
    defaultMeta: { service: 'exact-word-definer' }, // Add service name to all logs
    transports: [
        new transports.File({ // Log to a rotating file
            filename: 'logs/error.log',
            level: 'error', // Log only errors to this file
            maxsize: 5242880, // 5MB
            maxFiles: 5
        }),
        new transports.File({
            filename: 'logs/combined.log', // Log all levels (info, debug, etc.) to this file
            maxsize: 10485760, // 10MB
            maxFiles: 10
        })
    ]
});

// If not in production, add a console transport with pretty printing
if (process.env.NODE_ENV !== 'production') {
    logger.add(new transports.Console({
        format: format.combine(
            format.colorize(),
            format.printf(({ timestamp, level, message, reqId, ...meta }) => {
                const metaString = Object.keys(meta).length ? JSON.stringify(meta, null, 2) : ''; // Pretty print meta
                return `${timestamp} [${level}] ${reqId ? `[${reqId}]` : ''} ${message} ${metaString}`;
            })
        )
    }));
} else {
    // In production, log JSON to console as well (e.g., for log aggregators)
    logger.add(new transports.Console({
        format: format.json()
    }));
}

const app = express();

// --- Middleware Setup ---

// Add Request ID
app.use((req, res, next) => {
    req.id = req.headers['x-request-id'] || uuidv4();
    res.setHeader('X-Request-Id', req.id);
    // Add reqId to logger context for subsequent logs in this request
    req.logger = logger.child({ reqId: req.id });
    req.logger.info(`Incoming request: ${req.method} ${req.originalUrl}`, {
        httpMethod: req.method,
        url: req.originalUrl,
        ip: req.ip,
        headers: req.headers // Be cautious logging all headers in prod if sensitive info present
    });
    next();
});


// Security Headers
app.use(helmet());

// Enable CORS
app.use(cors()); // Consider restrictive origins in production

// Request Body Parsing
app.use(express.json());

// Basic Rate Limiting
const limiter = rateLimit({
    windowMs: RATE_LIMIT_WINDOW_MINUTES * 60 * 1000,
    max: RATE_LIMIT_MAX_REQUESTS,
    // message: { error: 'Too many requests, please try again later.'}, // Simplified message, handler provides detailed response
    standardHeaders: 'draft-7', // Recommended setting
    legacyHeaders: false,
    keyGenerator: (req) => {
        const word = req.params?.word || req.body?.word || req.query?.word || 'general';
        // Use req.ip which is more reliable behind proxies if 'trust proxy' is set
        return `${req.ip}|${word}`;
    },
    handler: (req, res, /*next, options*/) => {
        req.logger.warn('Rate limit exceeded', { ip: req.ip });
        res.status(429).json({
            error: 'Too many requests, please try again later.',
            requestId: req.id,
            retryAfterSeconds: Math.ceil(RATE_LIMIT_WINDOW_MINUTES * 60 / RATE_LIMIT_MAX_REQUESTS) // Estimate when a slot might be free
        });
    }
});
// If behind a proxy (like Nginx, Heroku), trust the first hop
// app.set('trust proxy', 1); // Uncomment if needed


// --- Metrics Store --- (Remains in-memory for this example)
const metrics = {
    totalRequests: 0,
    requestsSucceededExact: 0,
    requestsSucceededBestEffort: 0,
    requestsFailedInputValidation: 0,
    requestsFailedApiErrors: 0,
    requestsFailedServerErrors: 0,
    apiCallErrors: 0,
    totalWordsRequested: 0,
    totalWordsGenerated: 0,
    lastUpdated: null,
    startTime: new Date().toISOString(),
};

// --- OpenAI Client Initialization ---
const openai = new OpenAI({
    apiKey: OPENAI_API_KEY,
    baseURL: OPENAI_BASE_URL,
    timeout: 25000, // Slightly increased timeout
    maxRetries: 2
});

logger.info(`OpenAI Client Initialized`, { model: OPENAI_MODEL, baseURL: OPENAI_BASE_URL });

// --- Helper Functions ---
function countWords(text) {
    if (!text || typeof text !== 'string') {
        return 0;
    }
    const words = text.trim().split(/\s+/).filter(word => word.length > 0);
    return words.length;
}

// --- API Logic ---
async function handleDefineRequest(params, req, res) {
    const { logger: reqLogger, id: requestId } = req; // Use request-specific logger
    metrics.totalRequests++;
    metrics.lastUpdated = new Date().toISOString();
    let attemptHistoryForThisRequest = [];
    let previousDefinitions = [];
    let bestAttemptSoFar = null;
    let bestDifference = Infinity;
    let currentTemperature = 0.2;
    let cumulativeApiErrors = 0;

    try {
        const { word, length } = params;

        // Input Validation
        if (!word || typeof word !== 'string' || word.trim().length === 0) {
            metrics.requestsFailedInputValidation++;
            reqLogger.warn('Invalid input: "word" parameter missing or empty', { params });
            return res.status(400).json({ error: 'Invalid input. "word" must be a non-empty string.', requestId });
        }
        const requestedLength = parseInt(length, 10);
        // **FIXED**: Error message matches validation limit
        if (isNaN(requestedLength) || requestedLength <= 0 || requestedLength > 100) {
            metrics.requestsFailedInputValidation++;
            reqLogger.warn('Invalid input: "length" parameter invalid', { lengthParam: length, requestedLength, params });
            return res.status(400).json({ error: 'Invalid input. "length" must be a positive integer (1 to 250).', requestId });
        }

        const term = word.trim();
        reqLogger.info(`Processing request for definition`, { term, requestedLength });

        let attemptsLeft = MAX_RETRIES;

        while (attemptsLeft > 0) {
            const attemptNumber = MAX_RETRIES - attemptsLeft + 1;
            reqLogger.debug(`Attempt ${attemptNumber}/${MAX_RETRIES}`, { term, requestedLength, currentTemperature: currentTemperature.toFixed(3) });

            const previousAttemptsText = attemptHistoryForThisRequest
                .map((a, i) => `Attempt ${i + 1} Result (${a.actualWords} words, Target: ${requestedLength}):\n${a.output}`)
                .join('\n\n---\n');

            // Update prompt construction:
            const prompt = `Define "${term}" in EXACTLY ${requestedLength} words using this template:
            [${requestedLength} words] = [Context (20%)] + [Core Definition (60%)] + [Significance (20%)]
            
            REVISION INSTRUCTIONS:
            - If previous attempt was SHORT by 1-2 words: Add one precise adjective or clarifying phrase
            - If previous attempt was LONG by 1-2 words: Remove redundant adverbs or parentheticals
            - Never sacrifice accuracy for word count
            
            EXAMPLE FOR 100 WORDS:
            "${term}" refers to [concise 20-word context]. [60-word core definition with key details]. [20-word significance statement].
            
            CURRENT ATTEMPTS:
            ${previousAttemptsText || 'First attempt'}
            
            RESPONSE MUST:
            - Be exactly ${requestedLength} words
            - Use complete sentences
            - Avoid quotation marks
            - Exclude introductory phrases`;


            try {
                const startTime = Date.now();
                const response = await openai.chat.completions.create({
                    model: OPENAI_MODEL,
                    messages: [
                        { role: 'system', content: `You are a precise language assistant. Generate text definitions with an EXACT word count (${requestedLength} words). Strictly adhere to the word count. Analyze previous attempts if provided. Respond ONLY with the definition text.` },
                        { role: 'user', content: prompt }
                    ],
                    temperature: currentTemperature,
                    max_tokens: Math.max(200, requestedLength * 4), // Increased buffer slightly
                });
                reqLogger.info(`LLM Request Sent`, { term, response, requestedLength, attempt: attemptNumber, temperature: currentTemperature.toFixed(3) });
                console.log(`===========================`);
                console.log(response.choices[0].message.content)
                console.log(`===========================`);

                const durationMs = Date.now() - startTime;

                const outputRaw = response.choices[0]?.message?.content || '';
                let output = outputRaw.trim();
                if ((output.startsWith('"') && output.endsWith('"')) || (output.startsWith("'") && output.endsWith("'"))) {
                    output = output.substring(1, output.length - 1).trim();
                }
                const actualWords = countWords(output);
                const usage = response.usage; // Get token usage

                reqLogger.debug(`LLM Response Received`, {
                    attempt: attemptNumber,
                    actualWords,
                    durationMs,
                    modelUsed: response.model, // Log the actual model used by the API
                    tokenUsage: usage,
                    outputPreview: `"${output.substring(0, 70)}${output.length > 70 ? '...' : ''}"`
                });


                if (actualWords === 0 && output.length > 0) {
                    reqLogger.warn(`Word count is 0 but output exists`, { output });
                }

                const attemptData = {
                    attempt: attemptNumber,
                    temperature: currentTemperature,
                    output: output,
                    actualWords: actualWords,
                    targetWords: requestedLength,
                    timestamp: new Date().toISOString(),
                    durationMs,
                    tokenUsage: usage
                };
                attemptHistoryForThisRequest.push(attemptData);
                if (output) {
                    previousDefinitions.push(output);
                }

                const difference = Math.abs(actualWords - requestedLength);
                if ((actualWords > 0 || requestedLength === 0) && difference < bestDifference) {
                    bestDifference = difference;
                    bestAttemptSoFar = attemptData;
                    reqLogger.debug(`New best attempt found`, { attempt: attemptNumber, difference });
                }

                if (actualWords === requestedLength) {
                    reqLogger.info(`Exact word count achieved`, { term, requestedLength, attemptsMade: attemptNumber });
                    metrics.requestsSucceededExact++;
                    metrics.totalWordsGenerated += actualWords;
                    metrics.totalWordsRequested += requestedLength;
                    res.json({
                        result: output,
                        word: term,
                        requestedLength: requestedLength,
                        actualLength: actualWords,
                        status: "Exact length achieved",
                        attemptsMade: attemptNumber,
                        attemptsHistory: attemptHistoryForThisRequest.map(a => ({ // Keep history in response
                            attempt: a.attempt,
                            actualWords: a.actualWords,
                            difference: a.actualWords - requestedLength,
                            temp: parseFloat(a.temperature.toFixed(3))
                        })),
                        finalTemperature: parseFloat(currentTemperature.toFixed(3)),
                        tokenUsage: usage,
                        durationMs: attemptHistoryForThisRequest.reduce((sum, a) => sum + (a.durationMs || 0), 0),
                        requestId: requestId
                    });
                    return; // Success exit
                }

                // Modify temperature adjustment section:
                if (attemptsLeft > 1) {
                    const error = actualWords - requestedLength;
                    const errorAbs = Math.abs(error);
                    
                    // Enhanced scaling with minimum adjustment
                    let adjustment = Math.sign(error) * Math.max(
                        Math.min(errorAbs * 0.4, 0.15), // More aggressive scaling
                        0.05 // Minimum adjustment per attempt
                    );
                    
                    // Increase momentum impact
                    if (attemptHistoryForThisRequest.length > 1) {
                        const prevError = attemptHistoryForThisRequest[attemptHistoryForThisRequest.length - 2].actualWords - requestedLength;
                        adjustment += prevError * 0.15 * DECAY_FACTOR;
                    }
                    
                    currentTemperature = Math.min(0.7, Math.max(0.1, 
                        currentTemperature - adjustment // More responsive to error direction
                    ));
                    
                    // Add exploratory jitter
                    currentTemperature += (Math.random() - 0.5) * 0.03;
                    
                    // Apply snapping after calculation
                    currentTemperature = Math.round(currentTemperature * 20) / 20;
                }

            } catch (err) {
                cumulativeApiErrors++;
                metrics.apiCallErrors++;
                reqLogger.error(`OpenAI API error on attempt ${attemptNumber}`, {
                    error: {
                        message: err.message,
                        status: err.status,
                        type: err.type,
                        code: err.code,
                        // data: err.response?.data // Often contains useful details, but can be verbose
                    },
                    attempt: attemptNumber,
                    model: OPENAI_MODEL
                });

                if (err.status === 429 || err.message?.toLowerCase().includes('rate limit')) {
                    reqLogger.warn(`API Rate limit hit (status ${err.status || 'N/A'}). Waiting before retry...`);
                    if (OPENAI_BASE_URL.includes('pollinations')) {
                        reqLogger.warn('Pollinations rate limit likely hit. Consider Special Bee program.');
                    }
                    await new Promise(resolve => setTimeout(resolve, 1500 + Math.random() * 1000)); // Wait 1.5-2.5s
                } else if (err.status === 500 || err.status === 503) {
                    reqLogger.warn(`API Server error (${err.status}). Waiting before retry...`);
                    await new Promise(resolve => setTimeout(resolve, 2000 + Math.random() * 1000)); // Wait 2-3s
                } else if (err.code === 'ETIMEDEOUT' || err.code === 'ECONNABORTED') {
                    reqLogger.warn(`API Timeout error (${err.code}). Waiting before retry...`);
                    await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 500)); // Wait 1-1.5s
                }
            } finally {
                attemptsLeft--;
            }
        } // End of while loop

        // Handle Failure Cases (After all retries)
        if (bestAttemptSoFar) {
            reqLogger.warn(`Exact word count not achieved after ${MAX_RETRIES} attempts. Returning best effort.`, { term, requestedLength, bestDifference });
            metrics.requestsSucceededBestEffort++;
            metrics.totalWordsGenerated += bestAttemptSoFar.actualWords;
            metrics.totalWordsRequested += requestedLength;
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
                    temperature: parseFloat(a.temperature.toFixed(3)),
                    tokensUsed: {
                        prompt: a.tokenUsage?.prompt_tokens || 0,
                        completion: a.tokenUsage?.completion_tokens || 0,
                        total: a.tokenUsage?.total_tokens || 0
                    },
                    processingTime: `${a.durationMs}ms`,
                    attemptPrompt: `Define "${term}" in exactly ${requestedLength} words. Previous attempts: ${a.attempt > 1 ? attemptHistoryForThisRequest.length - 1 : 0}`
                })),
                summaryMetrics: {
                    totalTokensUsed: attemptHistoryForThisRequest.reduce((sum, a) => sum + (a.tokenUsage?.total_tokens || 0), 0),
                    avgTokensPerAttempt: parseFloat((attemptHistoryForThisRequest.reduce((sum, a) => sum + (a.tokenUsage?.total_tokens || 0), 0) / MAX_RETRIES).toFixed(2)),
                    totalProcessingTime: `${attemptHistoryForThisRequest.reduce((sum, a) => sum + a.durationMs, 0)}ms`,
                    temperatureRange: {
                        start: parseFloat(attemptHistoryForThisRequest[0].temperature.toFixed(3)),
                        end: parseFloat(currentTemperature.toFixed(3))
                    }
                },
                finalTemperature: parseFloat(bestAttemptSoFar.temperature.toFixed(3)),
                requestId: requestId
            });
        } else {
            reqLogger.error(`All ${MAX_RETRIES} attempts failed for definition request. No result generated.`, { term, requestedLength, cumulativeApiErrors });
            metrics.requestsFailedApiErrors++;
            res.status(503).json({
                error: "Failed to generate definition after multiple attempts due to upstream API errors.",
                word: term,
                requestedLength: requestedLength,
                attemptsMade: MAX_RETRIES,
                apiErrorsInRequest: cumulativeApiErrors,
                requestId: requestId
            });
        }

    } catch (error) {
        reqLogger.error('Critical Server Error in handleDefineRequest', {
            error: { message: error.message, stack: error.stack },
            requestParams: params
        });
        metrics.requestsFailedServerErrors++;
        // Avoid double counting API errors if it looks like one bubbling up
        if (cumulativeApiErrors === 0 && (error.status || error.response)) {
            metrics.apiCallErrors++;
        }
        res.status(500).json({
            error: 'An internal server error occurred while processing the definition request.',
            requestId: requestId,
            timestamp: new Date().toISOString()
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
    // (Metrics calculation logic remains the same as previous version)
    const now = new Date();
    const uptimeSeconds = process.uptime();
    const totalCompletedAttempts = metrics.requestsSucceededExact + metrics.requestsSucceededBestEffort;
    const totalFailedAttempts = metrics.requestsFailedInputValidation + metrics.requestsFailedApiErrors + metrics.requestsFailedServerErrors;
    const totalProcessedRequests = totalCompletedAttempts + totalFailedAttempts;
    const successRateExact = totalProcessedRequests > 0 ? (metrics.requestsSucceededExact / totalProcessedRequests * 100) : 0;
    const successRateBestEffort = totalProcessedRequests > 0 ? (totalCompletedAttempts / totalProcessedRequests * 100) : 0;
    const avgWordsGenerated = totalCompletedAttempts > 0 ? (metrics.totalWordsGenerated / totalCompletedAttempts) : 0;
    const avgWordsRequested = totalCompletedAttempts > 0 ? (metrics.totalWordsRequested / totalCompletedAttempts) : 0;

    res.json({
        apiVersion: API_VERSION,
        serverStartTime: metrics.startTime,
        currentTime: now.toISOString(),
        uptime: `${uptimeSeconds.toFixed(1)}s`,
        memoryUsage: process.memoryUsage(),
        totalRequestsReceived: metrics.totalRequests,
        processedRequests: {
            total: totalProcessedRequests,
            succeededExact: metrics.requestsSucceededExact,
            succeededBestEffort: metrics.requestsSucceededBestEffort,
            failedInputValidation: metrics.requestsFailedInputValidation,
            failedApiErrors: metrics.requestsFailedApiErrors,
            failedServerErrors: metrics.requestsFailedServerErrors,
        },
        apiInteraction: {
            totalApiCallErrors: metrics.apiCallErrors,
        },
        wordCounts: {
            averageWordsRequestedOnSuccess: parseFloat(avgWordsRequested.toFixed(1)),
            averageWordsGeneratedOnSuccess: parseFloat(avgWordsGenerated.toFixed(1)),
            totalWordsRequestedOnSuccess: metrics.totalWordsRequested,
            totalWordsGeneratedOnSuccess: metrics.totalWordsGenerated,
        },
        successRates: {
            exactMatch: `${successRateExact.toFixed(1)}%`,
            includingBestEffort: `${successRateBestEffort.toFixed(1)}%`,
        },
        lastMetricUpdate: metrics.lastUpdated,
        config: {
            model: OPENAI_MODEL,
            baseURL: OPENAI_BASE_URL,
            maxRetriesPerRequest: MAX_RETRIES,
            rateLimitMaxRequests: RATE_LIMIT_MAX_REQUESTS,
            rateLimitWindowMinutes: RATE_LIMIT_WINDOW_MINUTES,
            logLevel: LOG_LEVEL
        }
    });
});

// --- Catch-all for 404 Not Found ---
app.use((req, res, next) => {
    req.logger.warn('Route not found', { url: req.originalUrl }); // Use request logger
    res.status(404).json({
        error: 'Not Found',
        message: `The requested path ${req.originalUrl} does not exist on this server.`,
        requestId: req.id,
        timestamp: new Date().toISOString()
    });
});

// --- Final Error Handler ---
app.use((err, req, res, next) => {
    // Ensure req.logger exists, fallback to global logger
    const errorLogger = req.logger || logger;
    const reqId = req.id || 'N/A';

    errorLogger.error('Unhandled error caught by final error handler', {
        reqId: reqId,
        error: { message: err.message, stack: err.stack, status: err.status }, // Log full error details
        url: req.originalUrl,
        method: req.method
    });

    if (res.headersSent) {
        return next(err);
    }

    res.status(err.status || 500).json({
        error: 'Internal Server Error',
        message: process.env.NODE_ENV === 'production' ? 'An unexpected error occurred.' : err.message,
        requestId: reqId,
        timestamp: new Date().toISOString()
    });
});

// --- Server Start ---
app.listen(PORT, () => {
    logger.info(`-----------------------------------------`);
    logger.info(` Exact Word Definer API Starting...`);
    logger.info(` Version: ${API_VERSION}`);
    logger.info(` Environment: ${process.env.NODE_ENV || 'development'}`);
    logger.info(` Log Level: ${LOG_LEVEL}`);
    logger.info(` Port: ${PORT}`);
    logger.info(` OpenAI Model: ${OPENAI_MODEL}`);
    logger.info(` OpenAI Base URL: ${OPENAI_BASE_URL}`);
    logger.info(` Max Retries: ${MAX_RETRIES}`);
    logger.info(` Rate Limit: ${RATE_LIMIT_MAX_REQUESTS} req / ${RATE_LIMIT_WINDOW_MINUTES} min per key`);
    logger.info(` Security Headers: Enabled`);
    logger.info(` CORS: Enabled`);
    logger.info(`-----------------------------------------`);
    metrics.startTime = new Date().toISOString();
});

// Handle unhandled promise rejections and uncaught exceptions
process.on('unhandledRejection', (reason, promise) => {
    logger.fatal('Unhandled Rejection at:', { promise, reason: reason?.message || reason });
    // Optionally exit process after fatal error
    // process.exit(1);
});

process.on('uncaughtException', (error) => {
    logger.fatal('Uncaught Exception:', { error: { message: error.message, stack: error.stack } });
    // Optionally exit process after fatal error
    // process.exit(1);
});