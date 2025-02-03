const CACHE_NAME = 'prax-ai-chat-v1';
const urlsToCache = [
    '/',
    '/index.html',
    '/prax-ai-chat.js',
    '/MindAI-192x192.png',
    '/MindAI-512x512.png'
];

self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => cache.addAll(urlsToCache))
    );
});

self.addEventListener('fetch', event => {
    event.respondWith(
        caches.match(event.request)
            .then(response => response || fetch(event.request))
    );
});