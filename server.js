require('dotenv').config();
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const { OpenAI } = require('openai');
const { createLogger, format, transports } = require('winston'); // Winston for logging
const { v4: uuidv4 } = require('uuid'); // For request IDs
// console is a global object, no need to require it.
// const { console } = require('inspector'); // REMOVE this line

// --- Configuration & Setup ---

const PORT = process.env.PORT || 3001;
// parseInt with radix 10 is important
const MAX_RETRIES = parseInt(process.env.MAX_RETRIES || '10', 10);
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
// Adjusted default URL for clarity if it's non-standard. Use OpenAI's official URL by default.
// If using Pollinations, set OPENAI_BASE_URL to 'https://text.pollinations.ai/openai?referrer=ExactWordDefinerBot' in your .env
const OPENAI_BASE_URL = process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1';
const OPENAI_MODEL = process.env.OPENAI_MODEL || 'gpt-4o-mini'; // Default to a modern, cost-effective model

const RATE_LIMIT_MAX_REQUESTS = parseInt(process.env.RATE_LIMIT_MAX_REQUESTS || '100', 10);
const RATE_LIMIT_WINDOW_MINUTES = parseInt(process.env.RATE_LIMIT_WINDOW_MINUTES || '15', 10);

const API_VERSION = "1.5.1"; // Minor version bump for fixes and tone logic
const LOG_LEVEL = process.env.LOG_LEVEL || (process.env.NODE_ENV === 'production' ? 'info' : 'debug');

// Decay factor for temperature momentum - how much previous error influences the current temp change
const DECAY_FACTOR = 0.9;

// Exit immediately if API key is missing - server cannot function without it
if (!OPENAI_API_KEY) {
    console.error("FATAL ERROR: OPENAI_API_KEY environment variable is not set."); // Keep initial console for fatal startup errors
    process.exit(1);
}

// --- Logger Setup ---
const logger = createLogger({
    level: LOG_LEVEL,
    format: format.combine(
        format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }), // Standard timestamp format
        format.errors({ stack: true }), // Log stack traces for errors
        format.splat(), // Allows for string interpolation (e.g., logger.info('User %s logged in', userId))
        format.json()   // Output logs as JSON for easier parsing by log aggregators
    ),
    defaultMeta: { service: 'exact-word-definer-api' }, // Add service name to all logs
    transports: [
        // Log errors to a rotating file
        new transports.File({
            filename: 'logs/error.log',
            level: 'error',
            maxsize: 5242880, // 5MB
            maxFiles: 5, // Rotate logs after 5 files
            tailable: true // Start reading from the end
        }),
        // Log all levels (info, debug, etc.) to another rotating file
        new transports.File({
            filename: 'logs/combined.log',
            maxsize: 10485760, // 10MB
            maxFiles: 10, // Rotate logs after 10 files
            tailable: true
        })
    ],
    // Don't exit on error
    exitOnError: false,
});

// If not in production, add a console transport with pretty printing
if (process.env.NODE_ENV !== 'production') {
    logger.add(new transports.Console({
        format: format.combine(
            format.colorize(), // Add colors to log levels
            // Custom printf format for console output
            format.printf(({ timestamp, level, message, reqId, stack, ...meta }) => {
                const reqIdString = reqId ? `[${reqId}] ` : '';
                const metaString = Object.keys(meta).length ? JSON.stringify(meta, null, 2) : ''; // Pretty print meta data
                const stackString = stack ? `\n${stack}` : ''; // Add stack trace for errors
                return `${timestamp} [${level}] ${reqIdString}${message}${metaString}${stackString}`;
            })
        ),
        level: LOG_LEVEL // Ensure console logs at the specified level
    }));
} else {
    // In production, log JSON to console as well (useful for Docker logs / log aggregators)
    logger.add(new transports.Console({
        format: format.json(),
        level: LOG_LEVEL // Ensure console logs at the specified level
    }));
}

const app = express();

// --- Middleware Setup ---

// Add Request ID to incoming requests and response headers
app.use((req, res, next) => {
    req.id = req.headers['x-request-id'] || uuidv4(); // Use client-provided ID or generate new one
    res.setHeader('X-Request-Id', req.id); // Return the request ID to the client
    // Create a child logger for each request, automatically including the reqId
    req.logger = logger.child({ reqId: req.id });

    // Log basic request info
    req.logger.info(`Incoming request: ${req.method} ${req.originalUrl}`, {
        httpMethod: req.method,
        url: req.originalUrl,
        ip: req.ip, // req.ip is populated by Express; requires 'trust proxy' if behind one
        // headers: req.headers // Be cautious logging all headers in production due to potential sensitive info
    });
    next();
});

// Security Headers - Helps protect against common web vulnerabilities
app.use(helmet());

// Enable CORS - Allows cross-origin requests. Restrict this in production if possible.
app.use(cors()); // Use { origin: 'your-allowed-origin.com' } for production

// Request Body Parsing - Parses JSON request bodies
app.use(express.json());

// If behind a proxy (like Nginx, Heroku, AWS ELB), uncomment this line
// to ensure req.ip gets the client's real IP from the X-Forwarded-For header.
// app.set('trust proxy', 1);

// Basic Rate Limiting middleware
const limiter = rateLimit({
    windowMs: RATE_LIMIT_WINDOW_MINUTES * 60 * 1000, // Window size in milliseconds
    max: RATE_LIMIT_MAX_REQUESTS, // Max requests per window per key
    standardHeaders: 'draft-7', // Recommended setting for rate limit headers
    legacyHeaders: false, // Disable X-RateLimit-* headers
    // Key generator based on IP address and optionally the word requested for finer granularity
    keyGenerator: (req) => {
        // Use req.ip for the client IP. Fallback to 'unknown' if not available.
        // Combine with the word for per-word limiting, or just use req.ip for global IP limit.
        const word = req.params?.word || req.body?.word || req.query?.word || 'general';
        return `${req.ip || 'unknown'}|${word}`;
    },
    // Custom handler for rate-limited requests
    handler: (req, res) => {
        req.logger.warn('Rate limit exceeded', { ip: req.ip });
        res.status(429).json({
            error: `Too many requests for this word/IP combination, please try again later. Limit is ${RATE_LIMIT_MAX_REQUESTS} requests per ${RATE_LIMIT_WINDOW_MINUTES} minutes.`,
            requestId: req.id,
            // Provide an estimate of when a request might succeed. This is a rough estimate.
            retryAfterSeconds: Math.ceil(RATE_LIMIT_WINDOW_MINUTES * 60 / RATE_LIMIT_MAX_REQUESTS)
        });
    }
});


// --- Metrics Store --- (In-memory, reset on server restart)
const metrics = {
    totalRequests: 0, // Total requests received by the API endpoint
    requestsSucceededExact: 0, // Requests where exact word count was achieved
    requestsSucceededBestEffort: 0, // Requests where best effort result was returned
    requestsFailedInputValidation: 0, // Requests failing validation before LLM calls
    requestsFailedApiErrors: 0, // Requests failed due to persistent upstream API issues after retries
    requestsFailedServerErrors: 0, // Requests failed due to internal server errors
    apiCallErrors: 0, // Total count of *individual* API call errors across all attempts and requests
    totalWordsRequested: 0, // Sum of requested lengths for successful (exact or best effort) requests
    totalWordsGenerated: 0, // Sum of actual lengths for successful (exact or best effort) requests
    requestsByTone: {}, // Counts requests per user-provided tone string
    requestsByContext: {}, // Counts requests per context ('legal', 'educational', 'none')
    lastUpdated: null, // Timestamp of the last metric update
    startTime: new Date().toISOString(), // Server start time
};

// --- OpenAI Client Initialization ---
const openai = new OpenAI({
    apiKey: OPENAI_API_KEY,
    baseURL: OPENAI_BASE_URL,
    timeout: 30000, // Timeout for a single API call in milliseconds
    maxRetries: 2 // Client-level retries for transient network issues before our loop takes over
});

logger.info(`OpenAI Client Initialized`, { model: OPENAI_MODEL, baseURL: OPENAI_BASE_URL, clientTimeout: openai.timeout, clientMaxRetries: openai.maxRetries });


// --- Helper Functions ---

// Count words in a string, handling various whitespace and edge cases
function countWords(text) {
    if (!text || typeof text !== 'string') {
        return 0;
    }
    // Trim leading/trailing whitespace, replace multiple spaces with single,
    // remove common list markers at the start, and split by whitespace.
    const cleanedText = text.trim().replace(/\s+/g, ' ').replace(/^(\*|-|\d+\.)\s+/, '');
    const words = cleanedText.split(/\s+/).filter(word => word.length > 0);
    return words.length;
}

// --- API Logic ---
async function handleDefineRequest(params, req, res) {
    // Use request-specific logger and ID passed from middleware
    const { logger: reqLogger, id: requestId } = req;

    // Increment total requests metric immediately
    metrics.totalRequests++;
    metrics.lastUpdated = new Date().toISOString();

    let attemptHistoryForThisRequest = []; // Store details of each LLM attempt
    let bestAttemptSoFar = null; // Store the attempt closest to the target length
    let bestDifference = Infinity; // Smallest difference found so far
    let currentTemperature = 0.2; // Start temperature low for initial focus
    let cumulativeApiErrors = 0; // Count API errors encountered *during this specific request's retry loop*

    try {
        // --- Parameter Parsing and Validation ---
        const { word } = params;
        const length = params.length ? parseInt(params.length, 10) : NaN; // Parse length immediately
        // Allow any string for tone, trim and lowercase for internal consistency and metrics
        const tone = (params.tone || 'neutral').trim().toLowerCase();
        // Validate and lowercase context
        const validContexts = ['legal', 'educational', undefined];
        const contextRaw = params.context ? params.context.trim().toLowerCase() : undefined;
        const context = validContexts.includes(contextRaw) ? contextRaw : undefined; // Use undefined if context is invalid

        // Validate 'word' parameter
        if (!word || typeof word !== 'string' || word.trim().length === 0) {
             metrics.requestsFailedInputValidation++;
             reqLogger.warn('Invalid input: "word" parameter missing or empty');
             // Return 400 Bad Request response
             return res.status(400).json({
                 error: 'Invalid input. "word" parameter is required.',
                 requestId: requestId
            });
        }
        const term = word.trim(); // Use the trimmed word

        // Validate 'length' parameter
        if (isNaN(length) || length <= 0 || length > 250) { // Assuming a max length of 250 words is reasonable
             metrics.requestsFailedInputValidation++;
             reqLogger.warn('Invalid input: "length" parameter invalid', { lengthParam: params.length, parsedLength: length, params });
             // Return 400 Bad Request response
             return res.status(400).json({
                 error: 'Invalid input. "length" must be a positive integer (1 to 250).',
                 requestId: requestId
             });
        }
        const requestedLength = length;

        // Log a warning if context was provided but invalid, but continue processing
        if (params.context && context === undefined) {
             reqLogger.warn('Invalid context parameter provided, ignoring.', { originalContext: params.context, validContexts });
        }

        // --- Metrics Update after successful input validation ---
        metrics.requestsByTone[tone] = (metrics.requestsByTone[tone] || 0) + 1;
        metrics.requestsByContext[context || 'none'] = (metrics.requestsByContext[context || 'none'] || 0) + 1;

        reqLogger.info(`Processing request:`, { term, requestedLength, tone, context });

        // --- Retry Logic Loop ---
        let attemptsLeft = MAX_RETRIES;

        while (attemptsLeft > 0) {
            const attemptNumber = MAX_RETRIES - attemptsLeft + 1;
            reqLogger.debug(`Attempt ${attemptNumber}/${MAX_RETRIES}`, { term, requestedLength, currentTemperature: currentTemperature.toFixed(3) });

            // --- Dynamic Prompt Construction ---
            // Provide previous attempts to the model for self-correction
            const previousAttemptsText = attemptHistoryForThisRequest
                .map((a, i) => `Attempt ${i + 1} Result (Actual words: ${a.actualWords}, Target: ${requestedLength}):\n"${a.output.replace(/"/g, '')}"`) // Use quotes around the output in the prompt to help the model distinguish
                .join('\n\n---\n');

            // Construct style and context instructions for the prompt
            const styleInstructionString = tone && tone !== 'neutral' ? `Adopt a ${tone} tone.` : 'Maintain a neutral tone.';

            let contextInstructions = '';
            if (context === 'legal') {
                contextInstructions = 'Tailor the definition for a legal context. Focus on statutory definitions, legal precedents, and legal implications.';
            } else if (context === 'educational') {
                 contextInstructions = 'Tailor the definition for an educational context. Prioritize pedagogical clarity, learning objectives, and accessibility for students.';
            }

            const prompt = `Define "${term}" in EXACTLY ${requestedLength} words.

            STYLE AND CONTEXT GUIDELINES:
            ${styleInstructionString}
            ${contextInstructions ? contextInstructions + '\n' : ''}

            REVISION INSTRUCTIONS based on previous attempts:
            - Analyze the "CURRENT ATTEMPTS" below.
            - If previous attempt was SHORT by 1-3 words: Add one precise adjective, clarifying phrase, or relevant detail to slightly increase length.
            - If previous attempt was LONG by 1-3 words: Remove redundant adverbs, parentheticals, or less critical details to slightly decrease length.
            - If the word count difference is larger, carefully restructure sentences or add/remove substantial points while maintaining accuracy.
            - NEVER sacrifice accuracy or clarity for word count.

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
                                         ` ${styleInstructionString}` + // Incorporate the tone instruction
                                         `${contextInstructions ? ' ' + contextInstructions : ''}` + // Incorporate the context instruction
                                         ` Analyze the user's prompt, paying close attention to revision instructions based on previous attempts. Your response must be the definition text only, exactly ${requestedLength} words long.`;
            // --- End Dynamic System Message Construction ---


            try {
                const startTime = Date.now();
                // Make the OpenAI API call
                const response = await openai.chat.completions.create({
                    model: OPENAI_MODEL,
                    messages: [
                        { role: 'system', content: systemMessageContent }, // Use dynamic system message
                        { role: 'user', content: prompt } // Use dynamic prompt
                    ],
                    temperature: currentTemperature, // Use the adjusted temperature for the attempt
                    // Set a generous max_tokens, relying on the word count instruction for length control
                    max_tokens: 4096, // Max tokens the model can generate in this response
                    top_p: 1, // Consider all tokens with probability <= 1.0 - useful with temperature for balanced sampling
                    frequency_penalty: 0, // No penalty for repeating tokens
                    presence_penalty: 0, // No penalty for using new tokens
                });
                const durationMs = Date.now() - startTime; // Duration of the API call

                // Extract and clean the output
                const outputRaw = response.choices[0]?.message?.content || '';
                let output = outputRaw.trim();
                // Clean up potential leading/trailing quotes or list markers from model output
                 if ((output.startsWith('"') && output.endsWith('"')) || (output.startsWith("'") && output.endsWith("'"))) {
                     output = output.substring(1, output.length - 1).trim();
                 }
                 // Remove potential leading list markers like "1. ", "- ", "* "
                 output = output.replace(/^(\*|-|\d+\.)\s+/, '').trim();


                const actualWords = countWords(output); // Count words in the cleaned output
                const usage = response.usage; // Get token usage details

                // Log details of the LLM response for this attempt
                reqLogger.debug(`LLM Response Received`, {
                    attempt: attemptNumber,
                    actualWords,
                    targetWords: requestedLength,
                    difference: actualWords - requestedLength,
                    durationMs,
                    modelUsed: response.model, // Log the actual model used by the API
                    tokenUsage: usage,
                    // Log a preview of the output, truncated for brevity
                    outputPreview: `"${output.substring(0, 100)}${output.length > 100 ? '...' : ''}"`
                });

                // Handle edge case: output exists but word count is 0 (e.g., punctuation only)
                if (actualWords === 0 && output.length > 0) {
                    reqLogger.warn(`Word count is 0 but output exists, check countWords logic or empty response formatting issue from model. Output: "${output}"`);
                }

                // Store details of the current attempt
                const attemptData = {
                    attempt: attemptNumber,
                    temperature: parseFloat(currentTemperature.toFixed(3)), // Store temperature used for this attempt
                    output: output,
                    actualWords: actualWords,
                    targetWords: requestedLength,
                    timestamp: new Date().toISOString(),
                    durationMs,
                    tokenUsage: usage,
                    // Optionally log the prompt used for this attempt (can make logs very large)
                    // promptUsed: prompt
                };
                attemptHistoryForThisRequest.push(attemptData); // Add to history for this request

                // Update best attempt if this one is closer to the target
                const difference = Math.abs(actualWords - requestedLength);
                // Only consider attempts that produced *some* words as potential "best effort" results
                if ((actualWords > 0 || requestedLength === 0) && difference < bestDifference) {
                    bestDifference = difference;
                    bestAttemptSoFar = attemptData;
                    reqLogger.debug(`New best attempt found`, { attempt: attemptNumber, difference });
                }

                // --- Success Condition ---
                if (actualWords === requestedLength) {
                    reqLogger.info(`Exact word count achieved`, { term, requestedLength, attemptsMade: attemptNumber });
                    metrics.requestsSucceededExact++; // Increment success metric
                    metrics.totalWordsGenerated += actualWords;
                    metrics.totalWordsRequested += requestedLength;
                    // Send successful response
                    res.json({
                        result: output,
                        word: term,
                        requestedLength: requestedLength,
                        actualLength: actualWords,
                        status: "Exact length achieved",
                        attemptsMade: attemptNumber,
                        attemptsHistory: attemptHistoryForThisRequest.map(a => ({ // Include history
                            attempt: a.attempt,
                            actualWords: a.actualWords,
                            difference: a.actualWords - requestedLength, // Difference from target
                            temperature: a.temperature, // Temperature used for this attempt
                            tokensUsed: a.tokenUsage,
                            durationMs: a.durationMs
                        })),
                         summaryMetrics: { // Summary across all attempts for this request
                             totalTokensUsed: attemptHistoryForThisRequest.reduce((sum, a) => sum + (a.tokenUsage?.total_tokens || 0), 0),
                             avgTokensPerAttempt: parseFloat((attemptHistoryForThisRequest.reduce((sum, a) => sum + (a.tokenUsage?.total_tokens || 0), 0) / attemptNumber).toFixed(2)), // Average over successful attempts
                             totalProcessingTime: `${attemptHistoryForThisRequest.reduce((sum, a) => sum + a.durationMs, 0)}ms`,
                             temperatureRange: { // Range of temperatures used
                                 start: parseFloat(attemptHistoryForThisRequest[0]?.temperature?.toFixed(3) || '0.2'),
                                 end: parseFloat(currentTemperature.toFixed(3)) // Final temperature reached
                             }
                         },
                        finalTemperature: parseFloat(currentTemperature.toFixed(3)), // Temperature after the last attempt
                        requestId: requestId,
                        config: { tone, context } // Include request configuration
                    });
                    return; // Exit the function after sending response
                }

                // --- Temperature Adjustment for next attempt (if not the last attempt) ---
                if (attemptsLeft > 1) {
                    const error = actualWords - requestedLength; // How far off the word count is

                    // Basic proportional adjustment based on error size
                    // Smaller errors lead to smaller temp changes, larger errors larger changes (up to a cap)
                    let adjustment = Math.sign(error) * Math.min(
                        Math.abs(error) * 0.05, // Scale factor
                        0.15 // Maximum basic adjustment per step
                    );

                    // Add momentum from the previous error
                    if (attemptHistoryForThisRequest.length > 1) {
                         const prevError = attemptHistoryForThisRequest[attemptHistoryForThisRequest.length - 2].actualWords - requestedLength;
                         // Momentum adds a portion of the previous adjustment direction, decayed over time
                         adjustment += (prevError * 0.08) * DECAY_FACTOR; // Smaller momentum factor
                    }

                    // Update temperature, clamping between 0.1 and 0.7 (typical range for creative tasks)
                    currentTemperature = Math.min(0.7, Math.max(0.1,
                        currentTemperature - adjustment // Subtract adjustment (sign handles direction)
                    ));

                    // Add a small amount of random jitter to explore slightly different generation paths
                    currentTemperature += (Math.random() - 0.5) * 0.05; // Random value between -0.025 and +0.025

                    // Re-clamp temperature after adding jitter
                    currentTemperature = Math.min(0.7, Math.max(0.1, currentTemperature));

                    // Optional: Snap temperature to 2 decimal places to reduce prompt caching misses by API
                    currentTemperature = parseFloat(currentTemperature.toFixed(2));

                    reqLogger.debug(`Temperature adjustment applied`, {
                        attempt: attemptNumber,
                        actualWords,
                        requestedLength,
                        error,
                        adjustment: adjustment.toFixed(4),
                        newTemp: currentTemperature.toFixed(3)
                    });

                } else {
                    // This is the last attempt, no further temperature adjustment is needed
                    reqLogger.debug(`Last attempt (${attemptNumber}/${MAX_RETRIES}), no temperature adjustment.`, { currentTemperature: currentTemperature.toFixed(3) });
                }

            } catch (err) {
                cumulativeApiErrors++; // Increment error count for THIS request
                metrics.apiCallErrors++; // Increment total API call error count

                reqLogger.error(`OpenAI API error on attempt ${attemptNumber}`, {
                    error: {
                        message: err.message,
                        status: err.status,
                        type: err.type,
                        code: err.code,
                        // data: err.response?.data // Uncomment to log API response body on error (can be verbose)
                    },
                    attempt: attemptNumber,
                    model: OPENAI_MODEL,
                    currentTemperature: currentTemperature.toFixed(3)
                });

                // --- Backoff and Retry Strategy based on Error Type ---
                if (err.status === 429 || err.message?.toLowerCase().includes('rate limit')) {
                    // Rate Limit: Wait and retry
                    reqLogger.warn(`API Rate limit hit (status ${err.status || 'N/A'}). Waiting before retry...`);
                     if (OPENAI_BASE_URL.includes('pollinations')) {
                         reqLogger.warn('Pollinations rate limit likely hit. Consider Special Bee program.');
                     }
                    await new Promise(resolve => setTimeout(resolve, 2000 + Math.random() * 1500)); // Wait 2-3.5 seconds
                } else if (err.status >= 500) { // Server errors (500, 502, 503, 504)
                    // Upstream Server Error: Wait and retry
                    reqLogger.warn(`API Server error (${err.status}). Waiting before retry...`);
                    await new Promise(resolve => setTimeout(resolve, 2500 + Math.random() * 2000)); // Wait 2.5-4.5 seconds
                } else if (err.code === 'ETIMEDEOUT' || err.code === 'ECONNABORTED') {
                    // Network/Timeout Error: Wait and retry
                    reqLogger.warn(`API Timeout/Network error (${err.code}). Waiting before retry...`);
                    await new Promise(resolve => setTimeout(resolve, 1500 + Math.random() * 1000)); // Wait 1.5-2.5 seconds
                 } else if (err.status === 401 || err.status === 403) {
                     // Fatal API Key/Permission Error: Cannot recover, stop retrying
                     reqLogger.fatal(`API Authentication/Permission Error (${err.status}). Aborting retries.`, { errorDetails: err.message });
                     attemptsLeft = 0; // Force exit retry loop
                     metrics.requestsFailedApiErrors++; // Count this as a request failure due to API issues
                 } else {
                     // Other client errors (400, 404 etc.) likely indicate a persistent issue with the request or model config. Stop retrying.
                     reqLogger.error(`Unrecoverable API Client Error (${err.status || err.code}). Aborting retries.`, { errorDetails: err.message });
                     attemptsLeft = 0; // Force exit retry loop
                      metrics.requestsFailedApiErrors++; // Count this as a request failure due to API issues
                 }
            } finally {
                attemptsLeft--; // Decrement attempts counter regardless of success or failure within the attempt
            }
        } // End of while (attemptsLeft > 0) loop

        // --- Handle Failure Cases (After all retries are exhausted) ---
        // If a best attempt was found and it produced some output words
        if (bestAttemptSoFar && bestAttemptSoFar.actualWords > 0) {
            reqLogger.warn(`Exact word count not achieved after ${MAX_RETRIES} attempts. Returning best effort result.`, {
                term,
                requestedLength,
                bestDifference,
                actualWords: bestAttemptSoFar.actualWords
            });
            metrics.requestsSucceededBestEffort++; // Increment best effort success metric
            metrics.totalWordsGenerated += bestAttemptSoFar.actualWords; // Add words generated
            metrics.totalWordsRequested += requestedLength; // Add words requested

            // Send best effort response
            res.json({
                result: bestAttemptSoFar.output, // The output from the best attempt
                word: term,
                requestedLength: requestedLength,
                actualLength: bestAttemptSoFar.actualWords,
                status: `Best effort result (Closest: ${bestAttemptSoFar.actualWords} words, Difference: ${bestDifference})`,
                attemptsMade: MAX_RETRIES, // Report maximum attempts were made
                attemptsHistory: attemptHistoryForThisRequest.map(a => ({ // Include full history
                    attempt: a.attempt,
                    actualWords: a.actualWords,
                    targetDifference: a.actualWords - requestedLength,
                    temperature: parseFloat(a.temperature.toFixed(3)),
                    tokensUsed: a.tokenUsage,
                    durationMs: a.durationMs,
                })),
                summaryMetrics: { // Summary across all attempts
                    totalTokensUsed: attemptHistoryForThisRequest.reduce((sum, a) => sum + (a.tokenUsage?.total_tokens || 0), 0),
                    avgTokensPerAttempt: parseFloat((attemptHistoryForThisRequest.reduce((sum, a) => sum + (a.tokenUsage?.total_tokens || 0), 0) / MAX_RETRIES).toFixed(2)),
                    totalProcessingTime: `${attemptHistoryForThisRequest.reduce((sum, a) => sum + a.durationMs, 0)}ms`,
                    temperatureRange: {
                         start: parseFloat(attemptHistoryForThisRequest[0]?.temperature?.toFixed(3) || '0.2'),
                         end: parseFloat(currentTemperature.toFixed(3)) // Final temperature after loop
                    }
                },
                finalTemperature: parseFloat(currentTemperature.toFixed(3)), // Final temperature of the loop
                requestId: requestId,
                config: { tone, context } // Include request configuration
            });
        } else {
            // No successful attempt generated any output, or all attempts failed critically
            reqLogger.error(`All ${MAX_RETRIES} attempts failed or produced empty output for definition request.`, {
                term,
                requestedLength,
                cumulativeApiErrors,
                totalAttemptsMade: attemptHistoryForThisRequest.length
            });
            // Classify as API failure if there were API errors, otherwise a general server failure
            // Although if no output was ever produced, it strongly suggests an API/Model issue
            metrics.requestsFailedApiErrors++; // Assuming inability to get any useful output indicates an API/Model problem

            // Send error response
            res.status(503).json({ // 503 Service Unavailable is appropriate for upstream issues
                error: "Failed to generate definition after multiple attempts. The AI model may be unable to meet the exact requirements or there are upstream API issues. Please check input or try again later.",
                word: term,
                requestedLength: requestedLength,
                attemptsMade: attemptHistoryForThisRequest.length, // Report actual attempts made
                apiErrorsDuringAttempts: cumulativeApiErrors, // Report how many API errors occurred
                requestId: requestId
            });
        }

    } catch (error) {
        // Catch any unexpected errors during the function execution itself (e.g., logic errors, uncaught exceptions within the try block)
        reqLogger.error('Critical Server Error in handleDefineRequest processing logic', {
            error: {
                message: error.message,
                stack: error.stack, // Log stack trace for debugging
                name: error.name
            },
            requestParams: params
        });
        metrics.requestsFailedServerErrors++; // Increment internal server error metric

        // Send 500 Internal Server Error response
        res.status(500).json({
            error: 'An internal server error occurred while processing the definition request.',
            requestId: requestId,
            timestamp: new Date().toISOString()
        });
    }
};


// --- API Endpoints ---

// POST endpoint for defining words (parameters in request body)
app.post('/api/define', limiter, (req, res) => {
    handleDefineRequest(req.body, req, res); // Pass request body as parameters
});

// GET endpoint for defining words (parameters in query string)
app.get('/api/define', limiter, (req, res) => {
    handleDefineRequest(req.query, req, res); // Pass query parameters
});

// --- Metrics Endpoint ---
// Provides server performance and usage statistics
// Consider protecting this endpoint in production (e.g., with authentication)
app.get('/api/metrics', (req, res) => {
    // Use request logger, fallback if not available (shouldn't happen with middleware)
    const reqLogger = req.logger || logger.child({ reqId: req.id || 'metrics' });
    reqLogger.debug('Metrics endpoint requested');

    const now = new Date();
    const uptimeSeconds = process.uptime(); // Node.js process uptime in seconds
    const totalCompletedRequests = metrics.requestsSucceededExact + metrics.requestsSucceededBestEffort;
    const totalFailedRequests = metrics.requestsFailedInputValidation + metrics.requestsFailedApiErrors + metrics.requestsFailedServerErrors;
    // Total requests that reached the handleDefineRequest logic (passed basic middleware)
    const totalProcessedRequests = totalCompletedRequests + totalFailedRequests;

    // Calculate success rates, handle division by zero
    const successRateExact = totalProcessedRequests > 0 ? (metrics.requestsSucceededExact / totalProcessedRequests * 100) : 0;
    const successRateBestEffort = totalProcessedRequests > 0 ? (totalCompletedRequests / totalProcessedRequests * 100) : 0;

    // Calculate average words, handle division by zero
    const avgWordsGenerated = totalCompletedRequests > 0 ? (metrics.totalWordsGenerated / totalCompletedRequests) : 0;
    const avgWordsRequested = totalCompletedRequests > 0 ? (metrics.totalWordsRequested / totalCompletedRequests) : 0;

    // Send metrics response
    res.json({
        apiVersion: API_VERSION,
        serverStartTime: metrics.startTime,
        currentTime: now.toISOString(),
        uptime: `${uptimeSeconds.toFixed(1)} seconds`,
        memoryUsage: process.memoryUsage(), // Memory usage details
        totalRequestsReceived: metrics.totalRequests, // Total requests hitting the endpoint
        processedRequestsSummary: { // Requests that were processed by handleDefineRequest
            total: totalProcessedRequests,
            succeededExact: metrics.requestsSucceededExact,
            succeededBestEffort: metrics.requestsSucceededBestEffort,
            failedInputValidation: metrics.requestsFailedInputValidation,
            failedApiErrors: metrics.requestsFailedApiErrors, // Failures due to upstream API issues after retries
            failedServerErrors: metrics.requestsFailedServerErrors, // Failures due to internal logic errors
        },
        apiInteraction: {
            totalIndividualApiCallErrors: metrics.apiCallErrors, // Sum of errors across *all* LLM attempts
            // Could add average attempts per request here if tracked in attemptHistory
        },
        wordCounts: {
            averageWordsRequestedOnSuccess: parseFloat(avgWordsRequested.toFixed(1)),
            averageWordsGeneratedOnSuccess: parseFloat(avgWordsGenerated.toFixed(1)),
            totalWordsRequestedOnSuccess: metrics.totalWordsRequested, // Total requested length for successful requests
            totalWordsGeneratedOnSuccess: metrics.totalWordsGenerated, // Total actual length for successful requests
        },
        successRates: {
            exactMatch: `${successRateExact.toFixed(1)}%`,
            includingBestEffort: `${successRateBestEffort.toFixed(1)}%`,
        },
        requestsBreakdown: { // Breakdown by tone and context
            byTone: metrics.requestsByTone,
            byContext: metrics.requestsByContext
        },
        lastMetricUpdate: metrics.lastUpdated,
        config: { // Report current configuration settings
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
app.use((req, res) => {
    // Use request logger, fallback if not available
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
// This catches any errors thrown by preceding middleware or route handlers
app.use((err, req, res, next) => {
    // Ensure req.logger and req.id exist, fallback to global logger if middleware failed
    const errorLogger = req.logger || logger.child({ reqId: req.id || 'N/A' });
    const reqId = req.id || 'N/A';

    // Log the error details
    errorLogger.error('Unhandled error caught by final error handler', {
        reqId: reqId,
        error: {
            message: err.message,
            stack: err.stack, // Log full stack trace for debugging
            status: err.status,
            name: err.name
        },
        url: req.originalUrl,
        method: req.method,
        ip: req.ip // Log IP for debugging
    });

    // If headers have already been sent, delegate to default Express error handler
    // (shouldn't happen with proper async error handling but good practice)
    if (res.headersSent) {
        return next(err);
    }

    // Send a generic 500 Internal Server Error response
    // In production, hide the error message for security unless it's a known client error status
    const statusCode = err.status || 500;
    const errorMessage = process.env.NODE_ENV === 'production' && statusCode >= 500
                         ? 'An unexpected server error occurred.' // Generic message for server errors in production
                         : err.message; // Use specific error message otherwise

    res.status(statusCode).json({
        error: statusCode === 404 ? 'Not Found' : (statusCode >= 500 ? 'Internal Server Error' : 'Client Error'), // More specific error type
        message: errorMessage,
        requestId: reqId,
        timestamp: new Date().toISOString()
    });
});

// --- Server Start ---
const server = app.listen(PORT, () => {
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
    // Warn if using default OpenAI base URL but model suggests Pollinations or vice versa
    if (OPENAI_BASE_URL.includes('api.openai.com') && (OPENAI_MODEL.includes('pollinations') || process.env.OPENAI_BASE_URL === undefined || process.env.OPENAI_BASE_URL === 'https://text.pollinations.ai/openai?referrer=ExactWordDefinerBot')) {
         logger.warn(`Config Warning: Using OpenAI base URL '${OPENAI_BASE_URL}' but model '${OPENAI_MODEL}' or environment suggests Pollinations. Ensure OPENAI_BASE_URL is set correctly for your provider.`);
    } else if (OPENAI_BASE_URL.includes('pollinations.ai') && !OPENAI_MODEL.includes('pollinations') && process.env.OPENAI_MODEL === undefined) {
         logger.warn(`Config Warning: Using Pollinations base URL '${OPENAI_BASE_URL}' but model '${OPENAI_MODEL}' or environment suggests OpenAI. Ensure OPENAI_MODEL is set correctly for your provider.`);
    }
    logger.info(`-----------------------------------------`);
    metrics.startTime = new Date().toISOString(); // Record the actual server start time
});

// --- Process Error Handling ---

// Handle unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
    logger.fatal('Unhandled Rejection at:', { promise, reason: reason?.message || reason, stack: reason?.stack });
    // Decide whether to exit based on severity. For a web server, logging and staying alive might be preferable unless it's truly unrecoverable state.
    // process.exit(1); // Consider exiting for serious unhandled rejections
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
    logger.fatal('Uncaught Exception:', { error: { message: error.message, stack: error.stack } });
    // Uncaught exceptions often leave the process in an unstable state. It's common practice to exit.
    // Give Winston a moment to log before exiting.
    logger.on('finish', () => {
        process.exit(1);
    });
    // If logging doesn't finish, force exit after a delay
    setTimeout(() => {
      process.exit(1);
    }, 500); // Give it 500ms
});

// Optional: Handle graceful shutdown signals (SIGINT, SIGTERM)
process.on('SIGINT', () => gracefulShutdown('SIGINT'));
process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));

function gracefulShutdown(signal) {
    logger.info(`Received ${signal}. Starting graceful shutdown.`);

    // Stop the server from accepting new connections
    server.close(async (err) => {
        if (err) {
            logger.error('Error during server close', { error: err.message, stack: err.stack });
            // Force exit if server close fails? Or just proceed?
        }
        logger.info('HTTP server closed.');

        // Add any other cleanup logic here (e.g., closing database connections)
        // await closeDatabaseConnections();

        // Allow outstanding requests to finish (Express handles this largely with server.close)
        // In a real app, you might track open connections manually if needed.

        logger.info('Graceful shutdown complete. Exiting.');
        process.exit(0);
    });

    // Optional: Force shutdown after a timeout if graceful shutdown is stuck
    setTimeout(() => {
        logger.error('Graceful shutdown timed out. Forcing exit.');
        process.exit(1);
    }, 10000); // 10 seconds timeout
}