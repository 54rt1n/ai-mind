const CACHE_NAME = 'prax-ai-chat-v1';
const urlsToCache = [
    '/',
    '/index.html',
    '/aim-192x192.png',
    '/-512x512.png'
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => cache.addAll(urlsToCache))
    );
});

self.addEventListener('fetch', (event) => {
    event.respondWith(
        caches.match(event.request)
            .then((response) => response || fetch(event.request))
    );
});