require('dotenv').config();
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const { OpenAI } = require('openai');
const { createLogger, format, transports } = require('winston');
const { v4: uuidv4 } = require('uuid');
const mongoose = require('mongoose'); // Mongoose for MongoDB
const LanguageDetect = require('languagedetect'); // Use languagedetect

// --- Configuration & Setup ---

const PORT = process.env.PORT || 3001;
const MAX_RETRIES = parseInt(process.env.MAX_RETRIES || '10', 10);
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_BASE_URL = process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1';
const OPENAI_MODEL = process.env.OPENAI_MODEL || 'gpt-4o-mini';

const RATE_LIMIT_MAX_REQUESTS = parseInt(process.env.RATE_LIMIT_MAX_REQUESTS || '100', 10);
const RATE_LIMIT_WINDOW_MINUTES = parseInt(process.env.RATE_LIMIT_WINDOW_MINUTES || '15', 10);

// Configurable temperature parameters
const INITIAL_TEMPERATURE = parseFloat(process.env.INITIAL_TEMPERATURE || '0.2');
const TEMP_ADJUSTMENT_BASE = parseFloat(process.env.TEMP_ADJUSTMENT_BASE || '0.05'); // Base adjustment per word diff
const TEMP_ADJUSTMENT_PREV = parseFloat(process.env.TEMP_ADJUSTMENT_PREV || '0.08'); // Adjustment based on previous error
const TEMP_DECAY_FACTOR = parseFloat(process.env.TEMP_DECAY_FACTOR || '0.9'); // Decay factor for previous error influence
const TEMP_RANDOM_FACTOR = parseFloat(process.env.TEMP_RANDOM_FACTOR || '0.05'); // Random temperature wiggle
const TEMP_MIN = parseFloat(process.env.TEMP_MIN || '0.1');
const TEMP_MAX = parseFloat(process.env.TEMP_MAX || '0.7');

const MAX_REQUESTED_LENGTH = parseInt(process.env.MAX_REQUESTED_LENGTH || '250', 10); // Max word count requested
const MIN_DETECTION_LENGTH = parseInt(process.env.MIN_DETECTION_LENGTH || '5', 10); // Increased min length for detection (languagedetect might need more)

const VALID_TONES = ['neutral', 'formal', 'informal', 'humorous', 'serious']; // Define valid tones
const VALID_CONTEXTS = ['legal', 'educational']; // Define valid contexts

const API_VERSION = "1.12.0"; // Version bump for changing language detection library
const LOG_LEVEL = process.env.LOG_LEVEL || (process.env.NODE_ENV === 'production' ? 'info' : 'debug');

// MongoDB Configuration
const MONGODB_URI = process.env.MONGODB_URI;

// Metrics Saving Configuration
const METRICS_SAVE_INTERVAL_MS = parseInt(process.env.METRICS_SAVE_INTERVAL_MS || '600000', 10); // Default to 10 minutes (600000 ms)

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
    exitOnError: false, // Don't exit on handled errors
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
    // Production logs to console as JSON for easier parsing by log aggregators
    logger.add(new transports.Console({
        format: format.json(),
        level: LOG_LEVEL
    }));
}

const app = express();

// Serve static files from the 'public/docs' directory at the '/api/docs' path
app.use('/api/docs', express.static('public/docs'));

// --- Middleware Setup ---

// Add request ID and logger
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
// app.set('trust proxy', 1); // Uncomment if behind a proxy (e.g., Nginx, Cloudflare)

// Rate Limiter
const limiter = rateLimit({
    windowMs: RATE_LIMIT_WINDOW_MINUTES * 60 * 1000,
    max: RATE_LIMIT_MAX_REQUESTS,
    standardHeaders: 'draft-7',
    legacyHeaders: false,
    keyGenerator: (req) => {
        // Key based on IP and the word being defined.
        // Note: Language, length, tone, context are *not* part of the rate limit key.
        const word = (req.params?.word || req.body?.word || req.query?.word || '').toLowerCase().trim();
        return `${req.ip || 'unknown'}|${word || 'general'}`; // Use 'general' if word is empty
    },
    handler: (req, res) => {
        req.logger.warn('Rate limit exceeded', {
            ip: req.ip,
            word: (req.params?.word || req.body?.word || req.query?.word || '').toLowerCase().trim()
        });
        res.status(429).json({
            error: `Too many requests for this word/IP combination, please try again later. Limit is ${RATE_LIMIT_MAX_REQUESTS} requests per ${RATE_LIMIT_WINDOW_MINUTES} minutes.`,
            requestId: req.id,
            retryAfterSeconds: Math.ceil(RATE_LIMIT_WINDOW_MINUTES * 60 / RATE_LIMIT_MAX_REQUESTS)
        });
    },
    // Optionally add `store` for distributed rate limiting if needed (e.g., Redis)
    // store: new RedisStore({ ... })
});


// --- Metrics Store ---
// --- Metrics Store ---
const metrics = {
    totalRequests: 0, // Total requests reaching handleDefineRequest (pre-validation)
    // NOTE: totalRequests is *before* validation, processedRequestsSummary.total is *after* validation.
    // This distinction is useful for understanding where requests fail.

    processedRequestsSummary: { // Requests that went through validation and definition logic
        total: 0, // Total requests processed by handleDefineRequest
        succeededExactApi: 0, // API requests where exact word count was achieved
        succeededBestEffortApi: 0, // API requests where best effort result was returned
        cacheHits: 0, // Requests successfully served directly from DB (Moved here for clarity in processed summary)
        failedInputValidation: 0, // Failed during validateRequestParams
        failedApiErrors: 0, // Failed after retries exhausted due to API issues
        failedServerErrors: 0, // Failed due to unexpected internal server errors (including validation fails and other logic errors)
    },

    // --- Initialize apiInteraction object here with nested cacheMetrics ---
    apiInteraction: { // Grouping API interaction and cache metrics
        totalIndividualApiCallErrors: 0, // Individual LLM call errors (Moved here from top level)
        cacheMetrics: { // Cache-related counters (NEW nested object)
             cacheMisses: 0, // Requests that required calling the API after DB lookup (Moved here)
             cacheWritesSuccess: 0, // Successful cache writes (Moved here)
             cacheWritesFailed: 0, // Failed cache writes (e.g., duplicate key) (Moved here)
        }
    },

    totalWordsRequested: 0, // Sum of requested lengths for requests processed by API (cache misses)
    totalWordsGenerated: 0, // Sum of *actual* lengths generated by API for successful (exact/best effort) requests
    totalWordsServedFromCache: 0, // Sum of actual lengths served from cache

    requestsByTone: {},
    requestsByContext: {},
    requestsByEffectiveLang: {}, // Tracks the language code actually used for the query/cache
    requestsByDetectedLang: {}, // New metric for tracking detected language (when not user-specified)

    lastUpdated: null, // Timestamp of the last metric update (any metric change)
    startTime: new Date().toISOString(), // Server start time
    dbConnectionStatus: 'disconnected', // MongoDB connection status
    metricsSavingStatus: 'disabled', // Status of periodic metrics saving
    lastMetricsSaveAttempt: null, // Timestamp of the last attempt to save metrics
    metricsSaveIntervalMs: METRICS_SAVE_INTERVAL_MS // Report the configured interval
};


// --- MongoDB Initialization and Schemas ---
let isDbConnected = false;
let metricsSaveIntervalId = null; // To store the interval ID for clearing

// Schema for storing cached definitions
const definitionSchema = new mongoose.Schema({
    word: { type: String, required: true, lowercase: true, trim: true },
    length: { type: Number, required: true, min: 1 }, // Stores the ACTUAL word count generated by the AI attempt
    tone: { type: String, required: true, lowercase: true, default: 'neutral' },
    context: { type: String, lowercase: true, default: null }, // Store null for no context
    lang: { type: String, required: true, lowercase: true, trim: true, default: 'und' }, // Store the effective language code (ISO 639-3)
    definition: { type: String, required: true },
    createdAt: { type: Date, default: Date.now },
});

// Compound unique index includes 'lang'
definitionSchema.index({ word: 1, length: 1, tone: 1, context: 1, lang: 1 }, { unique: true });

// --- TTL index removed as per user request ---
// definitionSchema.index({ createdAt: 1 }, { expireAfterSeconds: CACHE_TTL_DAYS * 24 * 60 * 60 });
logger.info(`MongoDB Definition cache TTL index is NOT configured in code. Documents will not expire automatically.`);

const Definition = mongoose.model('Definition', definitionSchema);


// --- NEW Schema for Metrics Snapshots ---
const metricsSnapshotSchema = new mongoose.Schema({
    timestamp: { type: Date, default: Date.now, required: true, index: true }, // Index for time-series queries
    totalRequestsReceived: { type: Number, default: 0 },
    processedRequestsSummary: {
        total: { type: Number, default: 0 },
        succeededExactApi: { type: Number, default: 0 },
        succeededBestEffortApi: { type: Number, default: 0 },
        cacheHits: { type: Number, default: 0 },
        failedInputValidation: { type: Number, default: 0 },
        failedApiErrors: { type: Number, default: 0 },
        failedServerErrors: { type: Number, default: 0 }
    },
    apiInteraction: {
        totalIndividualApiCallErrors: { type: Number, default: 0 },
        cacheMetrics: {
             // cacheHits counter is duplicated in processedRequestsSummary, use that one for consistency in snapshots
            cacheMisses: { type: Number, default: 0 },
            cacheWritesSuccess: { type: Number, default: 0 },
            cacheWritesFailed: { type: Number, default: 0 }
        }
    },
    wordCounts: {
        totalWordsRequestedOnApiSuccess: { type: Number, default: 0 },
        totalWordsGeneratedOnApiSuccess: { type: Number, default: 0 },
        totalWordsServedFromCache: { type: Number, default: 0 }
    },
    requestsBreakdown: {
        byTone: { type: Object, default: {} }, // Store key-value pairs (e.g., {'neutral': 10, 'formal': 5})
        byContext: { type: Object, default: {} },
        byEffectiveLang: { type: Object, default: {} },
        byDetectedLang: { type: Object, default: {} }
    },
    configSummary: { // Optional: Save key config details with snapshot for context
        apiVersion: { type: String },
        model: { type: String },
        cacheExpiryStatus: { type: String }, // e.g., "Disabled" or "Enabled (TTL X days)"
        metricsSaveIntervalMs: { type: Number }
    }
    // Note: No TTL index on this schema if you want to keep data permanently
});

const MetricsSnapshot = mongoose.model('MetricsSnapshot', metricsSnapshotSchema);
logger.info(`MongoDB Metrics Snapshot schema defined.`);


// --- Function to save current metrics snapshot ---
async function saveMetricsSnapshot() {
    metrics.lastMetricsSaveAttempt = new Date().toISOString(); // Update attempt timestamp immediately

    if (!isDbConnected || metrics.dbConnectionStatus !== 'connected') {
        logger.warn('Metrics save skipped: Database not connected.', { dbStatus: metrics.dbConnectionStatus });
        metrics.metricsSavingStatus = 'skipped (db disconnected)';
        return;
    }

    try {
        const snapshot = new MetricsSnapshot({
            timestamp: new Date(),
            totalRequestsReceived: metrics.totalRequests,
            processedRequestsSummary: {
                total: metrics.processedRequestsSummary.total,
                succeededExactApi: metrics.requestsSucceededExact,
                succeededBestEffortApi: metrics.requestsSucceededBestEffort,
                cacheHits: metrics.processedRequestsSummary.cacheHits, // Use the one from processedRequestsSummary
                failedInputValidation: metrics.requestsFailedInputValidation,
                failedApiErrors: metrics.requestsFailedApiErrors,
                failedServerErrors: metrics.requestsFailedServerErrors
            },
             apiInteraction: {
                 totalIndividualApiCallErrors: metrics.apiCallErrors,
                 cacheMetrics: {
                     cacheMisses: metrics.cacheMisses, // Use the one from apiInteraction.cacheMetrics
                     cacheWritesSuccess: metrics.cacheWritesSuccess,
                     cacheWritesFailed: metrics.cacheWritesFailed
                 }
             },
            wordCounts: {
                totalWordsRequestedOnApiSuccess: metrics.totalWordsRequested,
                totalWordsGeneratedOnApiSuccess: metrics.totalWordsGenerated,
                totalWordsServedFromCache: metrics.totalWordsServedFromCache
            },
            requestsBreakdown: {
                 // Copy the current state of the breakdown objects
                byTone: { ...metrics.requestsByTone },
                byContext: { ...metrics.requestsByContext },
                byEffectiveLang: { ...metrics.requestsByEffectiveLang },
                byDetectedLang: { ...metrics.requestsByDetectedLang }
            },
            configSummary: {
                apiVersion: API_VERSION,
                model: OPENAI_MODEL,
                 // Reflect the actual schema configuration status
                cacheExpiryStatus: 'Disabled (TTL index removed from schema)', // This reflects the code change for Definitions
                metricsSaveIntervalMs: METRICS_SAVE_INTERVAL_MS
            }
        });

        await snapshot.save();
        logger.debug('Metrics snapshot saved successfully.');
        metrics.metricsSavingStatus = 'active';

    } catch (saveError) {
        logger.error('Failed to save metrics snapshot:', {
             error: saveError.message,
             stack: saveError.stack,
             dbStatus: metrics.dbConnectionStatus
        });
        metrics.metricsSavingStatus = `error: ${saveError.message ? saveError.message.substring(0, 50) : 'unknown'}...`; // Store truncated error message, handle null message
        // Continue running the application, don't crash on save error
    }
}


async function connectDB() {
    if (!MONGODB_URI) {
        logger.warn("MONGODB_URI not set. Running without database caching or metrics saving.");
        metrics.dbConnectionStatus = 'disabled';
        metrics.metricsSavingStatus = 'disabled (db disabled)';
        isDbConnected = false; // Ensure flag is false
        return;
    }

    try {
        // Attempt connection with retries built into Mongoose driver
        await mongoose.connect(MONGODB_URI, {
             serverSelectionTimeoutMS: 5000, // Timeout after 5s for server selection
             connectTimeoutMS: 10000 // Timeout after 10s for initial connection
        });
        isDbConnected = true;
        metrics.dbConnectionStatus = 'connected';
        logger.info('MongoDB connected successfully.');

        // --- Start periodic metrics saving AFTER successful DB connection ---
        if (METRICS_SAVE_INTERVAL_MS > 0) {
             metricsSaveIntervalId = setInterval(saveMetricsSnapshot, METRICS_SAVE_INTERVAL_MS);
             logger.info(`Periodic metrics saving enabled, interval: ${METRICS_SAVE_INTERVAL_MS} ms`);
             metrics.metricsSavingStatus = 'scheduled'; // Initial status, will change to 'active' on first successful save
             // Trigger a save on startup after successful connection
             saveMetricsSnapshot();
        } else {
             logger.warn('METRICS_SAVE_INTERVAL_MS is 0 or less. Periodic metrics saving is disabled.');
             metrics.metricsSavingStatus = 'disabled (interval 0)';
        }


        // Mongoose connection events
        mongoose.connection.on('error', (err) => {
            logger.error('MongoDB connection error:', { error: err.message, stack: err.stack });
            isDbConnected = false;
            metrics.dbConnectionStatus = 'error';
             // Clear interval on DB error
            if (metricsSaveIntervalId) {
                clearInterval(metricsSaveIntervalId);
                metricsSaveIntervalId = null;
                logger.warn('Metrics save interval cleared due to DB error.');
                metrics.metricsSavingStatus = 'stopped (db error)';
            }
        });
        mongoose.connection.on('disconnected', () => {
            logger.warn('MongoDB disconnected. Attempting to reconnect...');
            isDbConnected = false;
            metrics.dbConnectionStatus = 'disconnected';
             // Clear interval on DB disconnect
            if (metricsSaveIntervalId) {
                 clearInterval(metricsSaveIntervalId);
                 metricsSaveIntervalId = null;
                 logger.warn('Metrics save interval cleared due to DB disconnect.');
                 metrics.metricsSavingStatus = 'stopped (db disconnected)';
            }
        });
        mongoose.connection.on('connected', () => {
             // If reconnecting, log and potentially restart interval (if it was cleared)
             if (!isDbConnected) {
                 logger.info('MongoDB reconnected.');
                 isDbConnected = true;
                 metrics.dbConnectionStatus = 'connected';
                 // Restart interval if needed and configured
                 if (METRICS_SAVE_INTERVAL_MS > 0 && !metricsSaveIntervalId) {
                     metricsSaveIntervalId = setInterval(saveMetricsSnapshot, METRICS_SAVE_INTERVAL_MS);
                     logger.info(`Periodic metrics saving restarted, interval: ${METRICS_SAVE_INTERVAL_MS} ms`);
                     metrics.metricsSavingStatus = 'scheduled (reconnected)';
                      // Trigger a save on successful reconnection
                     saveMetricsSnapshot();
                 }
             }
        });
        mongoose.connection.on('reconnectFailed', () => {
             logger.fatal('MongoDB reconnect failed after multiple attempts.');
             metrics.dbConnectionStatus = 'reconnect_failed';
             // Interval should already be cleared by 'disconnected' handler
        });


    } catch (error) {
        logger.error('Failed to connect to MongoDB on startup:', { error: error.message, stack: error.stack });
        isDbConnected = false;
        metrics.dbConnectionStatus = 'startup_failed';
        metrics.metricsSavingStatus = 'disabled (startup failed)';
        // Allow the server to start even if initial DB connection fails
    }
}

// Initialize the database connection
connectDB();


// --- OpenAI Client Initialization ---
const openai = new OpenAI({
    apiKey: OPENAI_API_KEY,
    baseURL: OPENAI_BASE_URL,
    timeout: 30000, // Increased timeout for potentially longer responses
    maxRetries: 2 // Client-level retries before our loop starts
});

logger.info(`OpenAI Client Initialized`, {
    model: OPENAI_MODEL,
    baseURL: OPENAI_BASE_URL,
    clientTimeout: openai.timeout,
    clientMaxRetries: openai.maxRetries
});

// --- Language Detector Initialization ---
const detector = new LanguageDetect();
logger.info('LanguageDetect initialized.');

// Basic mapping from languagedetect names to ISO 639-3 codes
// This map is not exhaustive and may need expansion based on your expected input languages.
// Defaults to 'eng' if detection or mapping fails.
const langNameToISO = {
     'english': 'eng',
     'spanish': 'spa',
     'french': 'fra',
     'german': 'deu',
     'italian': 'ita',
     'portuguese': 'por',
     'russian': 'rus',
     'chinese': 'zho', // Or specific variants like 'cmn' (Mandarin), 'yue' (Cantonese)
     'japanese': 'jpn',
     'korean': 'kor',
     'arabic': 'ara',
     'hindi': 'hin',
     // Add more mappings as needed
};


// --- Helper Functions ---

/**
 * Counts words in a string, cleaning up common formatting.
 * @param {string} text - The text to count words in.
 * @returns {number} - The word count.
 */
function countWords(text) {
    if (!text || typeof text !== 'string') {
        return 0;
    }
    // Remove leading list markers (*, -, 1.), trim, replace multiple spaces with single, split by space/whitespace
    const cleanedText = text.trim().replace(/^(\*|-|\d+\.)\s+/, '').replace(/\s+/g, ' ');
    const words = cleanedText.split(/\s+/).filter(word => word.length > 0);
    return words.length;
}

/**
 * Validates and parses request parameters.
 * @param {object} params - The request parameters (query or body).
 * @param {object} logger - The request logger instance.
 * @returns {object} - Validated and parsed parameters.
 * @throws {Error} - If validation fails.
 */
function validateRequestParams(params, logger) {
    const { word } = params;
    const length = params.length ? parseInt(params.length, 10) : NaN;
    const tone = (params.tone || 'neutral').trim().toLowerCase();
    const contextRaw = params.context ? params.context.trim().toLowerCase() : undefined;
    const context = VALID_CONTEXTS.includes(contextRaw) ? contextRaw : undefined;
    const requestedLangParam = (params.lang && params.lang.trim() !== '') ? params.lang.trim().toLowerCase() : null;

    if (!word || typeof word !== 'string' || word.trim().length === 0) {
        throw new Error('Invalid input. "word" parameter is required and cannot be empty.');
    }
    const term = word.trim(); // Use original term for detection, lowercased for DB key/metrics consistency later

    if (isNaN(length) || length <= 0 || length > MAX_REQUESTED_LENGTH) {
        logger.warn('Invalid input: "length" parameter invalid', { lengthParam: params.length, parsedLength: length });
        throw new Error(`Invalid input. "length" must be a positive integer (1 to ${MAX_REQUESTED_LENGTH}).`);
    }
    const requestedLength = length;

    if (params.tone && !VALID_TONES.includes(tone)) {
        logger.warn('Invalid tone parameter provided, defaulting to neutral.', { originalTone: params.tone, validTones: VALID_TONES });
         // Note: 'tone' is already defaulted to 'neutral' if params.tone is falsy
    } else if (!params.tone) {
         // Use default 'neutral'
    } else {
         // Use provided valid tone
    }


    if (params.context && context === undefined) {
        logger.warn('Invalid context parameter provided, ignoring.', { originalContext: params.context, validContexts: VALID_CONTEXTS });
    }
    // Log a warning if lang was provided but empty/whitespace
    if (params.lang && params.lang.trim() === '') {
        logger.warn('Empty "lang" parameter provided, attempting auto-detection.', { originalLang: params.lang });
    }

    return { term, requestedLength, tone, context, requestedLangParam };
}

/**
 * Determines the effective language for the query based on user input or detection using languagedetect.
 * @param {string} term - The term to define.
 * @param {string|null} requestedLangParam - The language parameter provided by the user, or null.
 * @param {object} reqLogger - The request logger instance.
 * @returns {{effectiveLang: string, detectedLang: string|null, detectionDetails: object|null}} - The language to use, the detected language (if detection happened), and raw detection details.
 */
function determineEffectiveLanguage(term, requestedLangParam, reqLogger) {
    let effectiveLang = 'eng'; // Default fallback (ISO 639-3)
    let detectedLangName = null; // Store detected language name by the library
    let detectionConfidence = null; // Store detection confidence
    let detectionDetails = null; // Store raw detection result

    if (requestedLangParam) {
        effectiveLang = requestedLangParam; // User explicitly requested a language
        reqLogger.debug(`Using requested language: ${effectiveLang}`);
    } else {
        // Attempt language detection if no lang parameter was provided
        try {
            // Use languagedetect to detect language
            // Only attempt detection if the term is long enough
            if (term.length >= MIN_DETECTION_LENGTH) {
                 detectionDetails = detector.detect(term, 1); // Get the top result
                 reqLogger.debug('Languagedetect raw result:', { term, detectionDetails });

                 if (detectionDetails && detectionDetails.length > 0) {
                     const [langName, confidence] = detectionDetails[0];
                     detectedLangName = langName.toLowerCase();
                     detectionConfidence = confidence;

                     // Map language name to ISO 639-3 code
                     const mappedLang = langNameToISO[detectedLangName];

                     if (mappedLang) {
                         effectiveLang = mappedLang; // Use mapped detected language
                         reqLogger.debug(`Language detected and mapped: ${detectedLangName} (${effectiveLang}) with confidence ${confidence.toFixed(2)}`, { term });
                     } else {
                         reqLogger.warn(`Detected language "${detectedLangName}" has no ISO 639-3 mapping. Defaulting to "eng".`, { term, confidence: confidence.toFixed(2) });
                         effectiveLang = 'eng'; // Fallback if mapping is missing
                     }
                 } else {
                     reqLogger.debug('Language detection returned no results. Defaulting to "eng".', { term });
                     effectiveLang = 'eng'; // Fallback if detection returns empty
                 }
             } else {
                 reqLogger.debug(`Term too short (${term.length} chars) for language detection (min ${MIN_DETECTION_LENGTH}). Defaulting to "eng".`, { term });
                 effectiveLang = 'eng';
             }
        } catch (detectionError) {
            reqLogger.error('Error during language detection, defaulting to "eng".', {
                term,
                error: detectionError.message,
                stack: detectionError.stack
            });
            effectiveLang = 'eng'; // Fallback to English on error
        }
    }

     // Update metric for the detected language (only if detection was attempted)
     if (!requestedLangParam) {
         const finalDetectedKey = detectedLangName ? detectedLangName : (term.length < MIN_DETECTION_LENGTH ? 'und_too_short' : 'und_no_result');
         metrics.requestsByDetectedLang[finalDetectedKey] = (metrics.requestsByDetectedLang[finalDetectedKey] || 0) + 1;
     }

    // Return the effective language and the details of the detection attempt
    return {
        effectiveLang,
        detectedLang: (requestedLangParam ? null : detectedLangName), // Only report detected name if user didn't override
        detectionDetails: (requestedLangParam ? null : detectionDetails) // Only report raw details if user didn't override
    };
}


/**
 * Looks up a definition in the cache.
 * @param {object} lookupKey - The key fields for cache lookup.
 * @param {object} logger - The request logger instance.
 * @returns {Promise<object|null>} - The cached definition document or null.
 */
async function lookupDefinitionCache(lookupKey, logger) {
    if (!isDbConnected) {
        logger.debug('Database not connected, skipping cache lookup.');
        return null;
    }

    try {
        const cachedDef = await Definition.findOne(lookupKey).lean();

        if (cachedDef) {
            logger.info('Cache hit!', { lookupKey, actualLength: cachedDef.length });
            metrics.processedRequestsSummary.cacheHits++; // Increment cache hits metric
            metrics.totalWordsServedFromCache += cachedDef.length;
            metrics.lastUpdated = new Date().toISOString(); // Update metrics timestamp
            return cachedDef;
        } else {
            logger.debug('Cache miss.', { lookupKey });
            metrics.apiInteraction.cacheMetrics.cacheMisses++; // Increment cache misses metric
             metrics.lastUpdated = new Date().toISOString(); // Update metrics timestamp
            return null;
        }
    } catch (dbError) {
        logger.error('Database lookup error:', { error: dbError.message, stack: dbError.stack, lookupKey });
        metrics.dbConnectionStatus = 'error'; // Indicate potential DB issue
        // Don't return error, just log and continue without cache
        return null;
    }
}

/**
 * Saves a generated definition attempt to the cache.
 * @param {object} definitionData - The data for the definition to save.
 * @param {object} logger - The request logger instance.
 * @returns {Promise<void>}
 */
async function saveDefinitionToCache(definitionData, logger) {
     if (!isDbConnected) {
         logger.debug('Database not connected, skipping cache write.');
         return;
     }
     // Only cache attempts that produced words
     if (definitionData.length <= 0) {
          logger.debug('Skipping cache write for 0-word output.');
          return;
     }

     try {
         // Attempt to create the document
         // Mongoose create handles duplicate keys automatically with the unique index
         const doc = await Definition.create(definitionData);
         logger.debug('Cached definition attempt.', { term: definitionData.word, actualLength: definitionData.length, tone: definitionData.tone, context: definitionData.context, lang: definitionData.lang });
         metrics.cacheWritesSuccess++; // Increment cache writes success metric
          metrics.lastUpdated = new Date().toISOString(); // Update metrics timestamp
     } catch (dbWriteError) {
         if (dbWriteError.code === 11000) { // Duplicate key error
             logger.debug('Attempted to cache definition, but a duplicate already existed for this word, actual length, tone, context, lang.', {
                 term: definitionData.word, actualLength: definitionData.length, tone: definitionData.tone, context: definitionData.context, lang: definitionData.lang, error: dbWriteError.message
            });
             metrics.cacheWritesFailed++; // Still count as a write attempt, just failed due to duplicate
             metrics.lastUpdated = new Date().toISOString(); // Update metrics timestamp
         } else {
             logger.error('Database write error (attempt cache):', {
                 error: dbWriteError.message, stack: dbWriteError.stack,
                 definitionData: { word: definitionData.word, length: definitionData.length, tone: definitionData.tone, context: definitionData.context, lang: definitionData.lang }
             });
             metrics.cacheWritesFailed++; // Count other write failures
             metrics.dbConnectionStatus = 'error'; // Indicate DB issue
             metrics.lastUpdated = new Date().toISOString(); // Update metrics timestamp
         }
     }
}

/**
 * Builds the AI prompt and system message based on request parameters and attempt history.
 * @param {object} params - Validated request parameters (term, requestedLength, tone, context).
 * @param {string} effectiveLang - The effective language code (ISO 639-3).
 * @param {Array<object>} attemptHistory - History of previous attempts.
 * @returns {{prompt: string, systemMessage: string}}
 */
function buildPromptAndSystemMessage(params, effectiveLang, attemptHistory) {
    const { term, requestedLength, tone, context } = params;

    const previousAttemptsText = attemptHistory
        .map((a, i) => `Attempt ${i + 1} Result (Actual words: ${a.actualWords}, Target: ${requestedLength}):\n"${a.output.replace(/"/g, '')}"`)
        .join('\n\n---\n');

    const styleInstructionString = tone && tone !== 'neutral' ? `Adopt a ${tone} tone.` : 'Maintain a neutral tone.';

    let contextInstructions = '';
    if (context === 'legal') {
        contextInstructions = 'Tailor the definition for a legal context. Focus on statutory definitions, legal precedents, and legal implications.';
    } else if (context === 'educational') {
         contextInstructions = 'Tailor the definition for an educational context. Prioritize pedagogical clarity, learning objectives, and accessibility for students.';
    }

    // Language Instruction based on effectiveLang (using the ISO 639-3 code)
    // We can't reliably use the input language directly if detection failed or wasn't attempted effectively.
    // Stick to the determined effectiveLang ISO code.
    const langInstructionString = effectiveLang === 'und' ? // 'und' could happen if mapping failed or was undetermined
                                   `Define the term "${term}" in the most appropriate language based on the term itself.` :
                                   `Define the term in ${effectiveLang}.`;


    const prompt = `Define "${term}" in EXACTLY ${requestedLength} words.

    ${langInstructionString}
    ${styleInstructionString}
    ${contextInstructions ? contextInstructions + '\n' : ''}

    REVISION INSTRUCTIONS based on previous attempts:
    - Analyze the "CURRENT ATTEMPTS" below.
    - Identify the word count difference in the last attempt (${attemptHistory.length > 0 ? attemptHistory[attemptHistory.length-1].actualWords : '?'} vs ${requestedLength}).
    - If the last attempt was SHORT by 1-3 words: Add one precise adjective, clarifying phrase, or relevant detail to slightly increase length.
    - If the last attempt was LONG by 1-3 words: Remove redundant adverbs, parentheticals, or less critical details to slightly decrease length.
    - If the word count difference is larger, carefully restructure sentences or add/remove substantial points while maintaining accuracy.
    - ALWAYS prioritize accuracy and clarity over strict word count.
    - Do not reference previous attempts in your final output.

    EXAMPLE STRUCTURE GUIDELINE (Adjust structure approximately to match these proportions):
    For a ${requestedLength} word definition:
    [~${Math.max(5, Math.round(requestedLength * 0.2))} words: concise context or introduction].
    [~${Math.max(10, Math.round(requestedLength * 0.6))} words: core definition with key details].
    [~${Math.max(5, requestedLength - Math.max(5, Math.round(requestedLength * 0.2)) - Math.max(10, Math.round(requestedLength * 0.6)))} words: significance statement or application].

    CURRENT ATTEMPTS (Analyze this for revisions, but do not repeat or refer to it in your final response):
    ${previousAttemptsText || 'No previous attempts provided yet.'}

    RESPONSE MUST:
    - Be exactly ${requestedLength} words.
    - Consist ONLY of the definition text.
    - Use complete sentences.
    - Avoid quotation marks around the entire definition.
    - Exclude introductory phrases like "Definition:", "Here is a definition:", "The definition is:", etc.`;


    const systemMessageContent = `You are a highly precise language assistant trained to generate text definitions that adhere strictly to a given EXACT word count (${requestedLength} words).` +
                                 ` ${langInstructionString}` + // Use the same language instruction as the prompt
                                 ` ${styleInstructionString}` +
                                 `${contextInstructions ? ' ' + contextInstructions : ''}` +
                                 ` Analyze the user's prompt, paying close attention to revision instructions based on previous attempts. Your response must be the definition text only, exactly ${requestedLength} words long, and in the requested language.`;

    return { prompt, systemMessage: systemMessageContent };
}

/**
 * Makes a single call to the OpenAI API.
 * @param {string} prompt - The user prompt.
 * @param {string} systemMessage - The system message.
 * @param {number} temperature - The temperature to use.
 * @param {object} logger - The request logger instance.
 * @returns {Promise<object>} - The OpenAI API response object.
 * @throws {Error} - If the API call fails.
 */
async function makeApiCall(prompt, systemMessage, temperature, logger) {
    const startTime = Date.now();
    try {
        const response = await openai.chat.completions.create({
            model: OPENAI_MODEL,
            messages: [
                { role: 'system', content: systemMessage },
                { role: 'user', content: prompt }
            ],
            temperature: temperature,
            max_tokens: 4096, // Allow sufficient token generation
            top_p: 1,
            frequency_penalty: 0,
            presence_penalty: 0,
        });
        const durationMs = Date.now() - startTime;
        return { response, durationMs };
    } catch (err) {
        const durationMs = Date.now() - startTime;
        logger.error(`OpenAI API error`, {
            error: { message: err.message, status: err.status, type: err.type, code: err.code },
            durationMs, model: OPENAI_MODEL, temperature: temperature.toFixed(3)
        });
        metrics.apiCallErrors++; // Count individual API call errors
        metrics.lastUpdated = new Date().toISOString(); // Update metrics timestamp
        throw err; // Re-throw to be caught by the retry loop
    }
}

/**
 * Processes the raw API response for a single attempt.
 * @param {object} apiResponse - The raw response from makeApiCall.
 * @param {number} requestedLength - The target word count.
 * @param {number} attemptNumber - The current attempt number.
 * @param {number} temperature - The temperature used for this attempt.
 * @param {number} durationMs - The duration of the API call in ms.
 * @param {object} logger - The request logger instance.
 * @returns {object} - Structured attempt data.
 */
function processApiAttempt(apiResponse, requestedLength, attemptNumber, temperature, durationMs, logger) {
    const outputRaw = apiResponse.response.choices[0]?.message?.content || '';
    let output = outputRaw.trim();
     // Clean up common formatting added by LLMs
     if ((output.startsWith('"') && output.endsWith('"')) || (output.startsWith("'") && output.endsWith("'"))) {
         output = output.substring(1, output.length - 1).trim();
     }
     output = output.replace(/^(\*|-|\d+\.)\s+/, '').trim();


    const actualWords = countWords(output);
    const usage = apiResponse.response.usage;

    logger.debug(`LLM Response Processed`, {
        attempt: attemptNumber,
        actualWords,
        targetWords: requestedLength,
        difference: actualWords - requestedLength,
        durationMs,
        modelUsed: apiResponse.response.model,
        tokenUsage: usage,
        outputPreview: `"${output.substring(0, 100)}${output.length > 100 ? '...' : ''}"`
    });

    if (actualWords === 0 && output.length > 0) {
         logger.warn(`Word count is 0 but output exists. Output: "${output}"`);
    }

    return {
        attempt: attemptNumber,
        temperature: parseFloat(temperature.toFixed(3)),
        output: output,
        actualWords: actualWords,
        targetWords: requestedLength,
        timestamp: new Date().toISOString(),
        durationMs: durationMs,
        tokenUsage: usage,
    };
}

/**
 * Runs the retry loop for calling the OpenAI API.
 * @param {object} params - Validated request parameters (term, requestedLength, tone, context).
 * @param {string} effectiveLang - The effective language code (ISO 639-3).
 * @param {object} reqLogger - The request logger instance.
 * @returns {Promise<{bestAttemptSoFar: object|null, attemptHistory: Array<object>, cumulativeApiErrors: number, finalTemperature: number}>}
 */
async function runApiRetryLoop(params, effectiveLang, reqLogger) {
    const { term, requestedLength } = params;
    let attemptHistoryForThisRequest = [];
    let bestAttemptSoFar = null;
    let bestDifference = Infinity;
    let currentTemperature = INITIAL_TEMPERATURE;
    let cumulativeApiErrors = 0;
    let attemptsLeft = MAX_RETRIES;

    while (attemptsLeft > 0) {
        const attemptNumber = MAX_RETRIES - attemptsLeft + 1;
        reqLogger.debug(`Attempt ${attemptNumber}/${MAX_RETRIES}`, { term, requestedLength, currentTemperature: currentTemperature.toFixed(3) });

        // Pass the effectiveLang (ISO 639-3 code) to buildPromptAndSystemMessage
        const { prompt, systemMessage } = buildPromptAndSystemMessage(params, effectiveLang, attemptHistoryForThisRequest);

        try {
            const { response, durationMs } = await makeApiCall(prompt, systemMessage, currentTemperature, reqLogger);
            const attemptData = processApiAttempt({ response }, requestedLength, attemptNumber, currentTemperature, durationMs, reqLogger);
            attemptHistoryForThisRequest.push(attemptData);

            // --- Cache Write (Every Attempt with Output) ---
            // Save every generated definition that has words, alongside its *actual* word count, tone, context, and effective language.
            if (attemptData.actualWords > 0) { // Only save attempts with words
                 await saveDefinitionToCache({
                     word: term.toLowerCase(),
                     length: attemptData.actualWords, // IMPORTANT: Store the ACTUAL length generated
                     tone: params.tone,
                     context: params.context || null,
                     lang: effectiveLang, // Store the effective language used for this attempt (ISO 639-3)
                     definition: attemptData.output,
                     createdAt: new Date(),
                 }, reqLogger);
             }


            const difference = Math.abs(attemptData.actualWords - requestedLength);
            // Update best attempt if this one is closer AND has generated output words (> 0)
            if (attemptData.actualWords > 0) {
                 if (difference < bestDifference) {
                     bestDifference = difference;
                     bestAttemptSoFar = attemptData;
                     reqLogger.debug(`New best attempt found`, { attempt: attemptNumber, difference, actualWords: attemptData.actualWords });
                 }
                 // If no best attempt found yet, but this attempt has output, make it the best so far
                 else if (!bestAttemptSoFar) {
                     bestDifference = difference; // Calculate difference for this attempt
                     bestAttemptSoFar = attemptData;
                     reqLogger.debug(`First attempt with output found`, { attempt: attemptNumber, actualWords: attemptData.actualWords });
                 }
            }


            // --- Success Condition within the loop ---
            if (attemptData.actualWords === requestedLength) {
                reqLogger.info(`Exact word count achieved`, { term, requestedLength, attemptsMade: attemptNumber, lang: effectiveLang });
                // Return success data immediately
                return {
                    success: true,
                    result: attemptData.output,
                    actualLength: attemptData.actualWords,
                    attemptsMade: attemptNumber,
                    attemptHistory: attemptHistoryForThisRequest,
                    cumulativeApiErrors: cumulativeApiErrors,
                    finalTemperature: parseFloat(currentTemperature.toFixed(3))
                };
            }

            // --- Temperature Adjustment ---
            if (attemptsLeft > 1) {
                 if (attemptData.actualWords > 0) { // Only adjust temperature meaningfully if output had words
                    const error = attemptData.actualWords - requestedLength;
                    let adjustment = Math.sign(error) * Math.min( Math.abs(error) * TEMP_ADJUSTMENT_BASE, 0.2 ); // Cap base adjustment
                    if (attemptHistoryForThisRequest.length > 1) {
                        const prevError = attemptHistoryForThisRequest[attemptHistoryForThisRequest.length - 2].actualWords - requestedLength;
                        adjustment += (prevError * TEMP_ADJUSTMENT_PREV) * TEMP_DECAY_FACTOR;
                    }
                    currentTemperature = Math.min(TEMP_MAX, Math.max(TEMP_MIN, currentTemperature - adjustment));
                     // Add a small random wiggle to break out of potential local minima
                    currentTemperature += (Math.random() - 0.5) * TEMP_RANDOM_FACTOR;
                    currentTemperature = Math.min(TEMP_MAX, Math.max(TEMP_MIN, currentTemperature));
                    currentTemperature = parseFloat(currentTemperature.toFixed(3)); // Keep precision for logging
                    reqLogger.debug(`Temperature adjustment applied`, { attempt: attemptNumber, actualWords: attemptData.actualWords, requestedLength, error, adjustment: adjustment.toFixed(4), newTemp: currentTemperature });
                 } else {
                     // If 0 words, slightly randomize temperature to try a different path
                      currentTemperature = Math.min(TEMP_MAX, Math.max(TEMP_MIN, currentTemperature + (Math.random() * (TEMP_RANDOM_FACTOR * 2) - TEMP_RANDOM_FACTOR)));
                      currentTemperature = parseFloat(currentTemperature.toFixed(3));
                       reqLogger.debug(`Temperature randomized slightly due to 0 word output`, { attempt: attemptNumber, newTemp: currentTemperature });
                 }
            } else {
                reqLogger.debug(`Last attempt (${attemptNumber}/${MAX_RETRIES}), no temperature adjustment.`, { currentTemperature: currentTemperature.toFixed(3) });
            }


        } catch (err) {
            cumulativeApiErrors++;
            // Error logging already happens in makeApiCall

            // --- Backoff ---
            if (err.status === 429 || err.message?.toLowerCase().includes('rate limit')) {
                reqLogger.warn(`API Rate limit hit (status ${err.status || 'N/A'}). Waiting...`);
                if (OPENAI_BASE_URL.includes('pollinations')) reqLogger.warn('Pollinations rate limit likely hit.'); // Specific warning for Pollinations
                await new Promise(resolve => setTimeout(resolve, 2000 + Math.random() * 1500)); // Wait longer for rate limits
            } else if (err.status >= 500 || err.code === 'ETIMEDEOUT' || err.code === 'ECONNABORTED' || err.code === 'ECONNRESET') {
                reqLogger.warn(`API recoverable error (${err.status || err.code}). Waiting...`);
                await new Promise(resolve => setTimeout(resolve, 2500 + Math.random() * 2000)); // Wait longer for server errors/timeouts
             } else if (err.status === 401 || err.status === 403 || err.status === 400) { // Catch potentially unrecoverable client errors too
                 reqLogger.fatal(`API Authentication/Permission/Client Error (${err.status || err.code}). Aborting retries.`, { errorDetails: err.message });
                 attemptsLeft = 0; // Abort loop immediately
             } else {
                 reqLogger.error(`Uncaught API Client Error (${err.status || err.code}). Aborting retries.`, { errorDetails: err.message });
                 attemptsLeft = 0; // Abort loop immediately
             }
        } finally {
            attemptsLeft--;
        }
    } // End of while loop

    // --- Return Best Effort Result (if loop finishes without exact match) ---
    reqLogger.warn(`Retry loop finished after ${MAX_RETRIES} attempts without exact match. Returning best effort result if available.`);
     return {
         success: false, // Indicates loop finished without exact match
         bestAttemptSoFar: bestAttemptSoFar,
         attemptHistory: attemptHistoryForThisRequest,
         cumulativeApiErrors: cumulativeApiErrors,
         finalTemperature: parseFloat(currentTemperature.toFixed(3))
     };
}


// --- API Logic - Main Handler ---
async function handleDefineRequest(params, req, res) {
    const { logger: reqLogger, id: requestId } = req;

    // Update metrics Counters
    metrics.totalRequests++; // Increment total requests received (pre-validation)
    metrics.lastUpdated = new Date().toISOString();

    let validatedParams;
    let effectiveLangInfo; // Contains effectiveLang, detectedLang, detectionDetails
    let cachedDef = null;

    try {
        // --- Parameter Parsing and Validation ---
        try {
             validatedParams = validateRequestParams(params, reqLogger);
        } catch (validationError) {
            metrics.processedRequestsSummary.total++; // Count validation failures in processed total
            metrics.processedRequestsSummary.failedInputValidation++; // Increment validation failures metric
             metrics.lastUpdated = new Date().toISOString(); // Update metrics timestamp
            reqLogger.warn(`Input validation failed: ${validationError.message}`, { inputParams: params });
            return res.status(400).json({
                error: validationError.message,
                requestId: requestId
           });
        }
        const { term, requestedLength, tone, context, requestedLangParam } = validatedParams;

        // --- Determine Effective Language (using languagedetect) ---
        effectiveLangInfo = determineEffectiveLanguage(term, requestedLangParam, reqLogger);
        const { effectiveLang, detectedLang, detectionDetails } = effectiveLangInfo; // detectedLang here is the *name* from languagedetect

        // --- Update Metrics based on final determined request parameters ---
        // These are updated AFTER successful validation and language determination
        metrics.requestsByTone[tone] = (metrics.requestsByTone[tone] || 0) + 1;
        metrics.requestsByContext[context || 'none'] = (metrics.requestsByContext[context || 'none'] || 0) + 1;
        metrics.requestsByEffectiveLang[effectiveLang] = (metrics.requestsByEffectiveLang[effectiveLang] || 0) + 1; // Metric for the language used in DB/AI (ISO 639-3)
        // Metric for detected language uses the name from languagedetect or specific failure keys
        // This metric was updated inside determineEffectiveLanguage
        metrics.lastUpdated = new Date().toISOString(); // Update metrics timestamp

        reqLogger.info(`Processing request:`, { term, requestedLength, tone, context: context || 'none', effectiveLang, detectedLangName: detectedLang, detectionRaw: detectionDetails });

        // --- Cache Lookup (Read) ---
        // Cache lookup key includes the *effective* language and requested length
        const cacheLookupKey = {
             word: term.toLowerCase(),
             length: requestedLength, // Lookup by requested length to find exact matches
             tone: tone,
             context: context || null,
             lang: effectiveLang // Lookup by the effective language (ISO 639-3)
        };
        cachedDef = await lookupDefinitionCache(cacheLookupKey, reqLogger);

        if (cachedDef) {
            // Return cached definition immediately
            // Metrics already updated in lookupDefinitionCache
            metrics.processedRequestsSummary.total++; // Count this as a processed request
            metrics.lastUpdated = new Date().toISOString(); // Update metrics timestamp

            return res.json({
                result: cachedDef.definition,
                word: cachedDef.word, // Return lowercased word from DB for consistency
                requestedLength: requestedLength,
                actualLength: cachedDef.length,
                status: "Cached result (Exact length achieved)",
                attemptsMade: 0, // No API attempts made
                attemptsHistory: [],
                 summaryMetrics: { totalTokensUsed: 0, avgTokensPerAttempt: 0, totalProcessingTime: "0ms", temperatureRange: { start: 0, end: 0 } },
                finalTemperature: 0,
                requestId: requestId,
                config: { tone, context: context || 'none', effectiveLang: effectiveLang, detectedLangName: detectedLang }, // Report effectiveLang (ISO) and original detectedLang (name) if applicable
                cacheHit: true,
                cachedAt: cachedDef.createdAt
            });
        }

        // --- Run API Retry Loop (Only if cache missed) ---
        reqLogger.debug('Cache miss. Proceeding to API retry loop.');
        metrics.totalWordsRequested += requestedLength; // Only count requested words if going to API
        metrics.lastUpdated = new Date().toISOString(); // Update metrics timestamp

        // Pass the effectiveLang (ISO 639-3 code) to the retry loop
        const apiResult = await runApiRetryLoop(validatedParams, effectiveLang, reqLogger);

        const attemptHistorySummary = apiResult.attemptHistory.map(a => ({
             attempt: a.attempt,
             actualWords: a.actualWords,
             targetDifference: a.actualWords - requestedLength,
             temperature: a.temperature,
             tokensUsed: a.tokenUsage,
             durationMs: a.durationMs
         }));

         const totalTokensUsed = apiResult.attemptHistory.reduce((sum, a) => sum + (a.tokenUsage?.total_tokens || 0), 0);
         const totalDurationMs = apiResult.attemptHistory.reduce((sum, a) => sum + a.durationMs, 0);
         const avgTokensPerAttempt = apiResult.attemptHistory.length > 0 ? parseFloat((totalTokensUsed / apiResult.attemptHistory.length).toFixed(2)) : 0;
         const minTemp = apiResult.attemptHistory.length > 0 ? apiResult.attemptHistory.reduce((min, a) => Math.min(min, a.temperature), INITIAL_TEMPERATURE) : INITIAL_TEMPERATURE;
         const maxTemp = apiResult.attemptHistory.length > 0 ? apiResult.attemptHistory.reduce((max, a) => Math.max(max, a.temperature), INITIAL_TEMPERATURE) : INITIAL_TEMPERATURE;


         const summaryMetrics = {
             totalTokensUsed: totalTokensUsed,
             avgTokensPerAttempt: avgTokensPerAttempt,
             totalProcessingTime: `${totalDurationMs}ms`,
              temperatureRange: { start: parseFloat(minTemp.toFixed(3)), end: parseFloat(maxTemp.toFixed(3)) }
         };

        // --- Update Metrics Counters based on API Result ---
         metrics.processedRequestsSummary.total++; // Count this as a processed request
         if (apiResult.success) { // Exact match found within the loop
             metrics.processedRequestsSummary.succeededExactApi++;
             metrics.totalWordsGenerated += apiResult.actualLength;
         } else if (apiResult.bestAttemptSoFar && apiResult.bestAttemptSoFar.actualWords > 0) { // No exact match, but found a best effort (> 0 words)
             metrics.processedRequestsSummary.succeededBestEffortApi++;
             metrics.totalWordsGenerated += apiResult.bestAttemptSoFar.actualWords;
         } else { // No attempt produced valid output after all retries
              // This case is handled below with a 503 status
             metrics.processedRequestsSummary.failedApiErrors++; // Count as API failure if no valid output
         }
         metrics.lastUpdated = new Date().toISOString(); // Update metrics timestamp


        // --- Handle Final Result from Retry Loop ---
        if (apiResult.success) {
            res.json({
                result: apiResult.result,
                word: term,
                requestedLength: requestedLength,
                actualLength: apiResult.actualLength,
                status: "Exact length achieved",
                attemptsMade: apiResult.attemptsMade,
                attemptsHistory: attemptHistorySummary,
                summaryMetrics: summaryMetrics,
                finalTemperature: apiResult.finalTemperature,
                requestId: requestId,
                config: { tone, context: context || 'none', effectiveLang: effectiveLang, detectedLangName: detectedLang }, // Report effectiveLang (ISO) and original detectedLang (name) if applicable
                cacheHit: false
            });

        } else if (apiResult.bestAttemptSoFar && apiResult.bestAttemptSoFar.actualWords > 0) {
             const bestDifference = Math.abs(apiResult.bestAttemptSoFar.actualWords - requestedLength);
             reqLogger.warn(`Exact word count not achieved after ${MAX_RETRIES} attempts. Returning best effort result.`, {
                 term, requestedLength, bestDifference, actualWords: apiResult.bestAttemptSoFar.actualWords, effectiveLang: effectiveLang
             });

             res.json({
                 result: apiResult.bestAttemptSoFar.output,
                 word: term,
                 requestedLength: requestedLength,
                 actualLength: apiResult.bestAttemptSoFar.actualWords,
                 status: `Best effort result (Closest: ${apiResult.bestAttemptSoFar.actualWords} words, Difference: ${bestDifference})`,
                 attemptsMade: MAX_RETRIES, // Report total attempts
                 attemptsHistory: attemptHistorySummary,
                 summaryMetrics: summaryMetrics,
                 finalTemperature: apiResult.finalTemperature,
                 requestId: requestId,
                 config: { tone, context: context || 'none', effectiveLang: effectiveLang, detectedLangName: detectedLang }, // Report effectiveLang (ISO) and original detectedLang (name) if applicable
                 cacheHit: false
             });

        } else { // No attempt produced valid output after all retries
             reqLogger.error(`All ${MAX_RETRIES} attempts failed or produced empty output for definition request.`, {
                 term, requestedLength, cumulativeApiErrors: apiResult.cumulativeApiErrors, totalAttemptsMade: apiResult.attemptHistory.length, effectiveLang: effectiveLang
             });
             // Metric already incremented above if no success/best effort result

             res.status(503).json({
                 error: "Failed to generate definition after multiple attempts. The AI model may be unable to meet the exact requirements or there are upstream API issues. Please check input or try again later.",
                 word: term,
                 requestedLength: requestedLength,
                 attemptsMade: apiResult.attemptHistory.length,
                 apiErrorsDuringAttempts: apiResult.cumulativeApiErrors,
                 requestId: requestId,
                 config: { tone, context: context || 'none', effectiveLang: effectiveLang } // Only effectiveLang (ISO) known here
             });
        }


    } catch (error) {
        // This catch block handles errors not caught within the retry loop (e.g., logic errors, validation fails before API)
        // Metrics for total, validation, server errors are now handled inside the try/catch blocks.
        // Ensure no double counting here.
         metrics.processedRequestsSummary.total++; // Count this request as processed, but failed
         metrics.processedRequestsSummary.failedServerErrors++; // Increment server errors metric
         metrics.lastUpdated = new Date().toISOString(); // Update metrics timestamp


        reqLogger.error('Critical Server Error in handleDefineRequest processing logic', {
            error: { message: error.message, stack: error.stack, name: error.name },
            requestParams: params,
             // Report known config info even on error
            config: {
                 tone: validatedParams?.tone || params.tone || 'N/A',
                 context: validatedParams?.context || params.context || 'N/A',
                 effectiveLang: effectiveLangInfo?.effectiveLang || 'N/A', // ISO code
                 detectedLangName: effectiveLangInfo?.detectedLang || 'N/A' // language name
             }
        });

        // Check if response headers were already sent (e.g., by validation error handler)
        if (res.headersSent) {
            // If headers sent, rely on the final error handler middleware
            return;
        }

        res.status(500).json({
            error: 'An internal server error occurred while processing the definition request.',
            requestId: requestId,
            timestamp: new Date().toISOString(),
            config: { // Provide best-effort config info
                tone: validatedParams?.tone || params.tone || 'N/A',
                context: validatedParams?.context || params.context || 'N/A',
                effectiveLang: effectiveLangInfo?.effectiveLang || 'N/A' // ISO code
            }
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
    const totalCompletedRequests = metrics.processedRequestsSummary.succeededExactApi + metrics.processedRequestsSummary.succeededBestEffortApi + metrics.processedRequestsSummary.cacheHits;
    const totalFailedRequests = metrics.processedRequestsSummary.failedInputValidation + metrics.processedRequestsSummary.failedApiErrors + metrics.processedRequestsSummary.failedServerErrors;
    // The total processed requests is already tracked in metrics.processedRequestsSummary.total by handleDefineRequest
    const totalProcessedRequests = metrics.processedRequestsSummary.total;


    // Calculate success rates based on total requests *processed* by the handler
    const successRateExactApi = totalProcessedRequests > 0 ? (metrics.processedRequestsSummary.succeededExactApi / totalProcessedRequests * 100) : 0;
    const successRateIncludingBestEffortAndCache = totalProcessedRequests > 0 ? (totalCompletedRequests / totalProcessedRequests * 100) : 0;

    // These averages/totals ONLY count requests that went to the API and succeeded (cache misses that resulted in exact or best effort)
    const nonCachedSuccessfulRequests = metrics.processedRequestsSummary.succeededExactApi + metrics.processedRequestsSummary.succeededBestEffertApi;
    const avgWordsRequestedOnApiSuccess = parseFloat((metrics.totalWordsRequested / (nonCachedSuccessfulRequests || 1)).toFixed(1)); // Avoid div by zero
    const avgWordsGeneratedOnApiSuccess = parseFloat((metrics.totalWordsGenerated / (nonCachedSuccessfulRequests || 1)).toFixed(1)); // Avoid div by zero


    res.json({
        apiVersion: API_VERSION,
        serverStartTime: metrics.startTime,
        currentTime: now.toISOString(),
        uptime: `${uptimeSeconds.toFixed(1)} seconds`,
        memoryUsage: process.memoryUsage(),
        totalRequestsReceived: metrics.totalRequests, // Total requests reaching handler (pre-validation)
        processedRequestsSummary: { // Requests that went through validation and definition logic
            total: metrics.processedRequestsSummary.total,
            succeededExactApi: metrics.processedRequestsSummary.succeededExactApi,
            succeededBestEffortApi: metrics.processedRequestsSummary.succeededBestEffortApi,
            cacheHits: metrics.processedRequestsSummary.cacheHits,
            failedInputValidation: metrics.processedRequestsSummary.failedInputValidation,
            failedApiErrors: metrics.processedRequestsSummary.failedApiErrors,
            failedServerErrors: metrics.processedRequestsSummary.failedServerErrors,
        },
        apiInteraction: {
            totalIndividualApiCallErrors: metrics.apiCallErrors, // Sum of errors across *all* LLM attempts
             cacheMetrics: { // Breakdown of cache interactions
                 // cacheHits: metrics.processedRequestsSummary.cacheHits, // Use the one from processedRequestsSummary as the source
                 cacheMisses: metrics.apiInteraction.cacheMetrics.cacheMisses,
                 cacheWritesSuccess: metrics.cacheWritesSuccess,
                 cacheWritesFailed: metrics.cacheWritesFailed,
                 dbConnectionStatus: metrics.dbConnectionStatus,
             },
             metricsSaving: { // Report status of periodic metrics saving
                 status: metrics.metricsSavingStatus,
                 intervalMs: metrics.metricsSaveIntervalMs,
                 lastAttempt: metrics.lastMetricsSaveAttempt,
                 enabled: METRICS_SAVE_INTERVAL_MS > 0 && MONGODB_URI != null // Report if saving is enabled by config
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
            byEffectiveLang: metrics.requestsByEffectiveLang, // The language used for the query/cache (ISO 639-3)
            byDetectedLang: metrics.requestsByDetectedLang, // The language detected by languagedetect (name or failure reason)
        },
        lastMetricUpdate: metrics.lastUpdated, // Timestamp of the last update to any metric
        config: { // Full config details as before
            model: OPENAI_MODEL,
            baseURL: OPENAI_BASE_URL,
            maxRetriesPerRequest: MAX_RETRIES,
            rateLimitMaxRequests: RATE_LIMIT_MAX_REQUESTS,
            rateLimitWindowMinutes: RATE_LIMIT_WINDOW_MINUTES,
            logLevel: LOG_LEVEL,
            database: MONGODB_URI ? 'MongoDB (Enabled)' : 'Disabled',
            cacheExpiry: 'Disabled (TTL index removed from schema)', // Explicitly state expiry is off for Definitions
            metricsSavingIntervalMs: METRICS_SAVE_INTERVAL_MS, // Report the configured saving interval
            temperatureControl: {
                 initial: INITIAL_TEMPERATURE,
                 min: TEMP_MIN,
                 max: TEMP_MAX,
                 adjustmentBase: TEMP_ADJUSTMENT_BASE,
                 adjustmentPrev: TEMP_ADJUSTMENT_PREV,
                 decayFactor: TEMP_DECAY_FACTOR,
                 randomFactor: TEMP_RANDOM_FACTOR
            },
            maxRequestedLength: MAX_REQUESTED_LENGTH,
            minDetectionLength: MIN_DETECTION_LENGTH, // Updated config name
            validTones: VALID_TONES,
            validContexts: VALID_CONTEXTS,
            languageDetectionLibrary: 'languagedetect' // Explicitly state the library
        }
    });
});


// --- Catch-all for 404 Not Found ---
app.use((req, res) => {
    const reqLogger = req.logger || logger.child({ reqId: req.id || 'N/A' });
    reqLogger.warn('Route not found', { url: req.originalUrl, method: req.method });

    // Even 404 is a "processed" request from the middleware perspective
     if (req.id) {
         metrics.processedRequestsSummary.total++;
         metrics.processedRequestsSummary.failedServerErrors++; // Could argue this is client error, but marking as server handled failure
         metrics.lastUpdated = new Date().toISOString();
     }


    res.status(404).json({
        error: 'Not Found',
        message: `The requested path ${req.originalUrl} does not exist on this server.`,
        requestId: req.id || 'N/A',
        timestamp: new Date().toISOString()
    });
});

// --- Final Error Handler Middleware ---
// This middleware catches any errors passed via next(err) or thrown in middleware/routes
app.use((err, req, res, next) => {
    const errorLogger = req.logger || logger.child({ reqId: req.id || 'N/A' });
    const reqId = req.id || 'N/A';

    errorLogger.error('Unhandled error caught by final error handler', {
        reqId: reqId,
        error: {
            message: err.message,
            stack: err.stack,
            status: err.status,
            name: err.name,
            code: err.code // Include error code if available
        },
        url: req.originalUrl,
        method: req.method,
        ip: req.ip
    });

     // Increment metrics for uncaught errors IF they weren't already counted by handleDefineRequest
     // handleDefineRequest already increments total/failedServerErrors for errors *within* its logic.
     // This block should primarily catch errors from middleware *before* handleDefineRequest or unexpected crashes.
     // It's tricky to get this perfect without complex state tracking.
     // Simplest approach: If res.headersSent is false, assume this request hasn't been fully counted yet by handleDefineRequest logic.
     if (req.id && !res.headersSent) {
         metrics.processedRequestsSummary.total++; // Count this as a processed request that failed
         metrics.processedRequestsSummary.failedServerErrors++;
         metrics.lastUpdated = new Date().toISOString(); // Update metrics timestamp
     }


    if (res.headersSent) {
        // If headers were already sent, delegate to the default Express error handler
        return next(err);
    }

    const statusCode = err.status || 500;
    // In production, mask generic 500 errors
    const errorMessage = process.env.NODE_ENV === 'production' && statusCode >= 500 && statusCode !== 404 && statusCode !== 429 // Only mask >=500, exclude specific client errors
                         ? 'An unexpected internal server error occurred.'
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
    logger.info(` Database: ${MONGODB_URI ? `MongoDB (Attempting connection to ${MONGODB_URI.replace(/:\/\/(.+?):(.+?)@/, '://***:***@')})` : 'Disabled (MONGODB_URI not set)'}`); // Mask credentials in log
    logger.info(` Cache Expiry (Definitions): Disabled (TTL index removed from schema)`); // Log explicitly that definition cache expiry is disabled
    logger.info(` Metrics Saving: ${METRICS_SAVE_INTERVAL_MS > 0 && MONGODB_URI ? `Enabled (Interval: ${METRICS_SAVE_INTERVAL_MS} ms)` : 'Disabled (See config)'}`); // Log metrics saving status config
    logger.info(` Temperature Control: Initial=${INITIAL_TEMPERATURE}, Min=${TEMP_MIN}, Max=${TEMP_MAX}`);
    logger.info(` Max Requested Length: ${MAX_REQUESTED_LENGTH}`);
    logger.info(` Language Detection: languagedetect (Min length: ${MIN_DETECTION_LENGTH})`);


    // Warn if using default OpenAI base URL but model suggests Pollinations or vice versa
    if (OPENAI_BASE_URL.includes('api.openai.com') && (OPENAI_MODEL.includes('pollinations') || process.env.OPENAI_BASE_URL === undefined)) {
         logger.warn(`Config Warning: Using OpenAI base URL '${OPENAI_BASE_URL}' but model '${OPENAI_MODEL}' or environment suggests Pollinations. Ensure OPENAI_BASE_URL is set correctly for your provider (e.g., to 'https://api.pollinations.ai/v1').`);
    } else if (OPENAI_BASE_URL.includes('pollinations.ai') && !OPENAI_MODEL.includes('pollinations') && process.env.OPENAI_MODEL === undefined) {
         logger.warn(`Config Warning: Using Pollinations base URL '${OPENAI_BASE_URL}' but model '${OPENAI_MODEL}' or environment suggests OpenAI. Ensure OPENAI_MODEL is set correctly for your provider (e.g., to 'gpt-4o-mini' for OpenAI or 'pollinations/anything-v4#' for Pollinations).`);
    }
    logger.info(`-----------------------------------------`);

    metrics.startTime = new Date().toISOString();
});

// --- Process Error Handling ---
process.on('unhandledRejection', (reason, promise) => {
    logger.fatal('Unhandled Rejection at:', { promise, reason: reason?.message || reason, stack: reason?.stack });
    // Decide if you want to exit or keep running. For stability, exiting is often preferred.
    gracefulShutdown('unhandledRejection'); // Consider graceful exit
    // Or just log and potentially let PM2/container restart
});

process.on('uncaughtException', (error) => {
    logger.fatal('Uncaught Exception:', { error: { message: error.message, stack: error.stack } });
    // Critical error, should exit and let a process manager restart
    gracefulShutdown('uncaughtException'); // Attempt graceful shutdown first
});

// Graceful shutdown signals
process.on('SIGINT', () => gracefulShutdown('SIGINT'));
process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));

async function gracefulShutdown(signal) {
    logger.info(`Received ${signal}. Starting graceful shutdown.`);

     // Clear the metrics saving interval
    if (metricsSaveIntervalId) {
        clearInterval(metricsSaveIntervalId);
        metricsSaveIntervalId = null;
        logger.info('Metrics save interval cleared.');
        metrics.metricsSavingStatus = 'stopped (shutdown)';
    }

    // Attempt a final metrics save on shutdown if DB is connected
     if (isDbConnected && metrics.dbConnectionStatus === 'connected') {
         logger.info('Attempting final metrics snapshot save before closing DB.');
         // Use a timeout so we don't block shutdown indefinitely if save fails
         const saveTimeout = setTimeout(() => {
             logger.warn('Final metrics save attempt timed out.');
         }, 5000); // 5 seconds timeout for final save

         try {
             await saveMetricsSnapshot();
             logger.info('Final metrics snapshot saved successfully.');
         } catch (saveErr) {
             logger.error('Failed to save final metrics snapshot:', { error: saveErr.message });
         } finally {
             clearTimeout(saveTimeout);
         }
     } else {
         logger.warn('Final metrics save skipped: Database not connected or in error state.');
     }


    // Stop the server from accepting new connections
    server.close((err) => {
        if (err) {
            logger.error('Error during HTTP server close', { error: err.message, stack: err.stack });
        }
        logger.info('HTTP server closed.');

        // Then close database connection if it's open
        if (isDbConnected) {
            mongoose.connection.close(false) // false means do not force close active connections (though ideally none are long-lived)
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

    // Force shutdown after a timeout to prevent hanging
    setTimeout(() => {
        logger.error('Graceful shutdown timed out. Forcing exit.');
        process.exit(1);
    }, 25000); // Increased timeout slightly (25 seconds) to allow for final save + server close
}