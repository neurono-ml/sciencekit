#!/usr/bin/env node
// Post-process the mdBook output: add per-page canonical/Open Graph URLs and
// generate sitemap.xml so Google and Bing can discover every chapter.

import { readdir, readFile, stat, writeFile } from 'node:fs/promises';
import { join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const BASE_URL = 'https://neurono-ml.github.io/sciencekit/';
const BOOK_DIR = fileURLToPath(new URL('../book', import.meta.url));
const SKIPPED_PAGES = new Set(['print.html', '404.html']);

async function listHtmlFiles(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await listHtmlFiles(path)));
    } else if (entry.name.endsWith('.html')) {
      files.push(path);
    }
  }
  return files;
}

function pageUrl(file) {
  const relativePath = relative(BOOK_DIR, file).split(sep).join('/');
  // GitHub Pages serves index.html at the directory URL; keep the canonical
  // form without the file name so both spellings collapse to one URL.
  const canonicalPath = relativePath.replace(/(^|\/)index\.html$/, '$1');
  return BASE_URL + canonicalPath;
}

function escapeXml(value) {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&apos;');
}

function injectMetadata(html, url) {
  const tags = [
    `<link rel="canonical" href="${url}">`,
    `<meta property="og:url" content="${url}">`
  ].join('\n        ');
  if (html.includes('rel="canonical"')) return html;
  return html.replace('</head>', `        ${tags}\n    </head>`);
}

const files = (await listHtmlFiles(BOOK_DIR)).filter(
  (file) => !SKIPPED_PAGES.has(file.split(sep).pop())
);

const urls = [];
for (const file of files) {
  const url = pageUrl(file);
  const html = await readFile(file, 'utf8');
  const updated = injectMetadata(html, url);
  if (updated !== html) await writeFile(file, updated);
  const { mtime } = await stat(file);
  urls.push({ loc: url, lastmod: mtime.toISOString().slice(0, 10) });
}

urls.sort((left, right) => left.loc.localeCompare(right.loc));

const sitemap = [
  '<?xml version="1.0" encoding="UTF-8"?>',
  '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
  ...urls.map(
    ({ loc, lastmod }) =>
      `  <url><loc>${escapeXml(loc)}</loc><lastmod>${lastmod}</lastmod></url>`
  ),
  '</urlset>',
  ''
].join('\n');

await writeFile(join(BOOK_DIR, 'sitemap.xml'), sitemap);
console.log(`postbuild: ${urls.length} pages indexed in sitemap.xml`);
