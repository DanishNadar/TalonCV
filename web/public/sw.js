const CACHE = "taloncv-runtime-v2";
const SHELL = ["/", "/dashboard/", "/interview/new/"];
const MODEL_HOSTS = ["storage.googleapis.com", "huggingface.co", "cdn-lfs.huggingface.co"];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then((cache) => cache.addAll(SHELL)).catch(() => undefined));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(caches.keys()
    .then((names) => Promise.all(names.filter((name) => name.startsWith("taloncv-runtime-") && name !== CACHE).map((name) => caches.delete(name))))
    .then(() => self.clients.claim()));
});

function store(request, response) {
  if (response.ok || response.type === "opaque") void caches.open(CACHE).then((cache) => cache.put(request, response.clone()));
  return response;
}

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;
  const url = new URL(event.request.url);
  const sameOrigin = url.origin === self.location.origin;
  if (!sameOrigin && !MODEL_HOSTS.includes(url.hostname) && !url.hostname.endsWith(".hf.co")) return;

  // Documents must stay network-first. A cached page keeps pointing at the build
  // hashes it shipped with, so serving it after a redeploy renders a blank app.
  if (event.request.mode === "navigate") {
    event.respondWith(fetch(event.request).then((response) => store(event.request, response)).catch(() => caches.match(event.request).then((cached) => cached || caches.match("/"))));
    return;
  }

  // Build assets and public model files are content-addressed or immutable, so
  // cache-first is safe and keeps a cached install usable offline.
  event.respondWith(caches.match(event.request).then((cached) => cached || fetch(event.request).then((response) => store(event.request, response))));
});
