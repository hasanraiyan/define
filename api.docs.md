# Exact Word Definer API Documentation

This document describes the API endpoints for the Exact Word Definer service, which uses AI to generate definitions of a specific word count.

**API Version:** `1.12.0` (Uses `languagedetect` for auto-detection)

## Base URL

The base URL for all API endpoints is typically:

`http://localhost:3001`

*Note: This may change depending on deployment environment.*

## Authentication

No API key or specific authentication is required for consuming the `/api/define` endpoint by default. Rate limiting and IP tracking are used for abuse prevention.

The `/api/metrics` endpoint is also currently unsecured.

## Endpoints

### 1. Define Word

Defines a word to a specific word count using an AI model, optionally adjusting tone and context.

*   **URL:** `/api/define`
*   **Methods:** `GET`, `POST`

    *   Use `GET` for simple requests where parameters fit comfortably in the URL query string.
    *   Use `POST` with a JSON body for requests, which is generally preferred and allows for potentially longer or more complex parameters in the future.

*   **Parameters:**

    | Name      | Type    | Required | Default     | Description                                                                 | Valid Values                       |
    | :-------- | :------ | :------- | :---------- | :-------------------------------------------------------------------------- | :--------------------------------- |
    | `word`    | `string`| Yes      | N/A         | The word or term to define. Trimmed before processing.                    | Non-empty string                   |
    | `length`  | `integer`| Yes      | N/A         | The **target** number of words for the definition. Must be positive.      | `1` to `MAX_REQUESTED_LENGTH` (`250` by default) |
    | `tone`    | `string`| No       | `neutral`   | The desired tone for the definition. Case-insensitive.                    | `neutral`, `formal`, `informal`, `humorous`, `serious` (others ignored) |
    | `context` | `string`| No       | `null`      | The context for the definition (e.g., tailoring for a specific domain).   | `legal`, `educational` (others ignored) |
    | `lang`    | `string`| No       | Auto-detect | The desired language (ISO 639-3 code, e.g., `eng`, `spa`, `fra`). If omitted or empty, language is auto-detected from the `word`. Case-insensitive. | ISO 639-3 codes (e.g., 'eng', 'spa') |

*   **Example Request (GET):**

    ```bash
    GET /api/define?word=serendipity&length=30&tone=neutral&lang=eng HTTP/1.1
    Host: localhost:3001
    ```

*   **Example Request (POST):**

    ```bash
    POST /api/define HTTP/1.1
    Host: localhost:3001
    Content-Type: application/json

    {
      "word": "ephemeral",
      "length": 25,
      "tone": "formal",
      "context": null,
      "lang": "eng"
    }
    ```

*   **Response (Success - Status Code `200 OK`):**

    Returns a JSON object containing the definition and details about the request processing.

    ```json
    {
      "result": "A definition of the word to the requested length and style...",
      "word": "ephemeral",
      "requestedLength": 25,
      "actualLength": 25,
      "status": "Exact length achieved", // or "Best effort result (Closest: X words, Difference: Y)", or "Cached result (Exact length achieved)"
      "attemptsMade": 1, // Number of API calls made for this request (0 if cached)
      "attemptsHistory": [ // Array detailing each AI attempt (included for debugging/advanced users)
        {
          "attempt": 1,
          "temperature": 0.200,
          "output": "...", // The raw output text
          "actualWords": 25,
          "targetWords": 25,
          "timestamp": "YYYY-MM-DDTHH:mm:ss.SSSZ",
          "durationMs": 1234,
          "tokenUsage": { /* ... OpenAI token usage details */ }
        }
      ],
      "summaryMetrics": { // Summary metrics for the API calls made for this request
        "totalTokensUsed": 150,
        "avgTokensPerAttempt": 150.00,
        "totalProcessingTime": "1234ms",
        "temperatureRange": { "start": 0.200, "end": 0.200 } // Range of temperatures used
      },
      "finalTemperature": 0.200, // Temperature of the last attempt
      "requestId": "a1b2c3d4-e5f6-7890-1234-567890abcdef", // Unique ID for tracing
      "config": { // Configuration used for this specific request
        "tone": "formal",
        "context": "none", // or "legal", "educational"
        "effectiveLang": "eng", // The ISO 639-3 code used for the AI query/cache
        "detectedLangName": null // Name reported by detector if `lang` was omitted (e.g., "english", "spanish")
      },
      "cacheHit": false, // true if result was served from cache
      "cachedAt": null // Timestamp if cacheHit is true
    }
    ```

*   **Response (Error):**

    Returns a JSON object describing the error. Status code varies depending on the error type (e.g., 400, 429, 500, 503).

    ```json
    {
      "error": "Error Type Summary", // e.g., "Bad Request", "Too Many Requests", "Internal Server Error", "Service Unavailable"
      "message": "More specific error details...", // e.g., "Invalid input. 'length' must be...", "Too many requests...", "Failed to generate definition after..."
      "requestId": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
      "timestamp": "YYYY-MM-DDTHH:mm:ss.SSSZ", // Present for server/unhandled errors
      "word": "invalid", // Present for input validation errors
      "requestedLength": 0, // Present for input validation errors
      "attemptsMade": 10, // Present for 503 errors
      "apiErrorsDuringAttempts": 5, // Present for 503 errors
      "config": { // Best-effort config info, may be incomplete on validation errors
          "tone": "neutral",
          "context": "none",
          "effectiveLang": "eng"
      }
    }
    ```

*   **Status Codes:**
    *   `200 OK`: Request was successful.
    *   `400 Bad Request`: Input validation failed (e.g., missing `word`, invalid `length`).
    *   `404 Not Found`: The requested endpoint path does not exist.
    *   `429 Too Many Requests`: Rate limit exceeded for the word/IP combination. Includes `retryAfterSeconds`.
    *   `500 Internal Server Error`: An unexpected server-side error occurred during processing.
    *   `503 Service Unavailable`: The API failed to generate a valid definition after all retry attempts (often due to persistent upstream AI errors or inability to meet constraints).

### 2. Service Metrics

Provides operational metrics and configuration details for the service.

*   **URL:** `/api/metrics`
*   **Method:** `GET`
*   **Parameters:** None.
*   **Authentication:** Currently unsecured.
*   **Response (Success - Status Code `200 OK`):**

    Returns a large JSON object with various statistics (total requests, success/failure counts, cache performance, word counts, breakdowns by tone/context/language) and configuration values.

    *Example (Truncated):*

    ```json
    {
      "apiVersion": "1.12.0",
      "serverStartTime": "YYYY-MM-DDTHH:mm:ss.SSSZ",
      "currentTime": "YYYY-MM-DDTHH:mm:ss.SSSZ",
      "uptime": "XYZ seconds",
      "memoryUsage": { /* ... details ... */ },
      "totalRequestsReceived": 100, // Total requests reaching handler
      "processedRequestsSummary": { /* ... breakdown by success/failure type ... */ },
      "apiInteraction": { /* ... total individual API call errors, cache stats ... */ },
      "wordCounts": { /* ... average/total words requested/generated ... */ },
      "successRates": { /* ... percentage rates ... */ },
      "requestsBreakdown": { // Metrics broken down by request parameters
          "byTone": { "neutral": 50, "formal": 30, /* ... */ },
          "byContext": { "none": 80, "legal": 10, /* ... */ },
          "byEffectiveLang": { "eng": 95, "spa": 5, /* ... */ }, // ISO code used for query/cache
          "byDetectedLang": { "english": 80, "spanish": 10, "und_too_short": 5, /* ... */ } // Name from detector or failure reason
      },
      "lastMetricUpdate": "YYYY-MM-DDTHH:mm:ss.SSSZ",
      "config": { /* ... current service configuration values ... */ },
      // ... potentially other metrics ...
    }
    ```

## Headers

*   **Request:** `Content-Type: application/json` should be included for `POST` requests.
*   **Response:** `X-Request-Id`: A unique ID generated for each incoming request, useful for tracing requests through logs.

## Rate Limiting

Requests to the `/api/define` endpoint are rate limited per IP address and the `word` parameter.

*   Limit: `RATE_LIMIT_MAX_REQUESTS` requests (`100` by default)
*   Window: `RATE_LIMIT_WINDOW_MINUTES` minutes (`15` by default)

If the limit is exceeded for a specific word/IP combination within the time window, the API will return a `429 Too Many Requests` status code with an error message. The response will include a `retryAfterSeconds` field suggesting how long to wait before retrying.

## Caching

The service utilizes a MongoDB cache to store previously generated definitions.

*   **Cache Key:** Definitions are cached based on the combination of `word` (lowercased), the **actual** `length` generated by the AI, `tone`, `context` (or `null`), and the `effectiveLang` (the ISO 639-3 code used for the query).
*   **Lookup:** A cache lookup is performed before calling the AI. A cache hit occurs only if an existing entry matches the requested `word`, `length`, `tone`, `context`, and `effectiveLang`.
*   **Writing:** Every AI attempt that produces a definition with words (`actualWords > 0`) is saved to the cache using its **actual** generated length as part of the key.
*   **TTL:** Cached entries expire after `CACHE_TTL_DAYS` days (`30` by default).
*   **Impact:** A cache hit bypasses the AI call, resulting in a much faster response time and reduced load on the AI provider. `cacheHit: true` will be included in the response JSON for cache hits.

## Language Detection

If the `lang` parameter is not provided in the request, the API attempts to automatically detect the language of the `word` using the `languagedetect` library.

*   Detection is attempted only if the `word` is at least `MIN_DETECTION_LENGTH` characters long (`5` by default).
*   The detected language name (e.g., "english", "spanish") is mapped to an ISO 639-3 code (e.g., "eng", "spa") to become the `effectiveLang` used for the AI prompt and caching.
*   If detection fails, returns an undetermined result (`und`), or the detected language has no known mapping, the `effectiveLang` defaults to `eng`.
*   The API response includes both the `effectiveLang` (ISO code used) and `detectedLangName` (the name from the detector if auto-detection occurred) for transparency.

## Error Handling Summary

*   `400 Bad Request`: Client-side input error (missing required parameters, invalid format/values).
*   `404 Not Found`: Invalid API endpoint path.
*   `429 Too Many Requests`: Rate limit exceeded.
*   `503 Service Unavailable`: Upstream AI service failure after multiple retries or inability to meet constraints.
*   `500 Internal Server Error`: Unhandled server-side error.

All error responses include a `requestId` for tracing.

## Configuration Notes

The behavior of the API is influenced by several environment variables:

*   `PORT`: HTTP server port.
*   `MAX_RETRIES`: Maximum number of AI call attempts per request to achieve the target length.
*   `OPENAI_API_KEY`: API key for the AI provider (required).
*   `OPENAI_BASE_URL`: Alternative URL for the AI provider (e.g., for proxy or other providers).
*   `OPENAI_MODEL`: The specific AI model to use.
*   `RATE_LIMIT_MAX_REQUESTS`, `RATE_LIMIT_WINDOW_MINUTES`: Rate limiting settings.
*   `INITIAL_TEMPERATURE`, `TEMP_ADJUSTMENT_BASE`, etc.: Parameters controlling the temperature adjustment logic during retries.
*   `MAX_REQUESTED_LENGTH`: Maximum allowed value for the `length` parameter.
*   `MIN_DETECTION_LENGTH`: Minimum word length for language detection.
*   `MONGODB_URI`: Connection string for the MongoDB cache (if not set, caching is disabled).
*   `CACHE_TTL_DAYS`: Time-to-live for cache entries.
*   `LOG_LEVEL`: Winston logging level (`debug`, `info`, `warn`, `error`, `fatal`).

