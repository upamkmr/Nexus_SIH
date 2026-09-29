/**
 * Lazy, fail-safe Redis client.
 * Nothing in the current codebase imports this yet, but it's here and ready.
 * Connection errors are caught and logged — they never block startup.
 */
const { REDIS_URL } = require('./env');

let _client = null;

function getRedisClient() {
  if (_client) return _client;

  try {
    // ioredis is an optional peer dep — gracefully skip if not installed
    const Redis = require('ioredis'); // eslint-disable-line
    _client = new Redis(REDIS_URL, {
      lazyConnect: true,
      enableOfflineQueue: false,
      maxRetriesPerRequest: 1,
      retryStrategy: (times) => (times > 3 ? null : Math.min(times * 200, 2000))
    });

    _client.on('connect', () => console.log('[Redis] Connected to', REDIS_URL));
    _client.on('error',   (err) => console.warn('[Redis] Connection error (non-fatal):', err.message));
    _client.connect().catch(() => {}); // non-blocking
  } catch (err) {
    console.warn('[Redis] ioredis not installed or failed to load — caching disabled:', err.message);
    // Return a no-op stub so callers don't need to null-check
    _client = {
      get:    async () => null,
      set:    async () => 'OK',
      del:    async () => 0,
      exists: async () => 0,
      quit:   async () => {}
    };
  }

  return _client;
}

module.exports = { getRedisClient };
