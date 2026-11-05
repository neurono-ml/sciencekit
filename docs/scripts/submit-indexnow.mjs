#!/usr/bin/env node
// Submit every documentation URL to IndexNow (Bing, Yandex, Seznam, Naver...).
// The key file must already be published at the site root (docs/src/<key>.txt).

const HOST = 'neurono-ml.github.io';
const BASE_URL = `https://${HOST}/sciencekit/`;
const KEY = '4f6c6bea947c13936cc169b2f9e34826';
const ENDPOINT = 'https://api.indexnow.org/indexnow';

const sitemap = await fetch(`${BASE_URL}sitemap.xml`).then((response) => {
  if (!response.ok) throw new Error(`sitemap request failed: ${response.status}`);
  return response.text();
});

const urlList = [...sitemap.matchAll(/<loc>([^<]+)<\/loc>/g)].map((match) =>
  match[1].replaceAll('&amp;', '&')
);

if (urlList.length === 0) {
  console.error('indexnow: no URLs found in sitemap.xml');
  process.exit(1);
}

const response = await fetch(ENDPOINT, {
  method: 'POST',
  headers: { 'content-type': 'application/json; charset=utf-8' },
  body: JSON.stringify({
    host: HOST,
    key: KEY,
    keyLocation: `${BASE_URL}${KEY}.txt`,
    urlList
  })
});

if (!response.ok && response.status !== 202) {
  console.error(`indexnow: submission failed with status ${response.status}`);
  process.exit(1);
}

console.log(`indexnow: submitted ${urlList.length} URLs (status ${response.status})`);
