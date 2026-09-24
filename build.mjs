#!/usr/bin/env node
'use strict';

// Build «Три невозможные вещи»: stitches the shell (markup + css + js) and the
// three scene modules into two static outputs. No dependencies, Node 22+.
//
//   node build.mjs                          -> ./index.html, ./dist/artifact.html
//   node build.mjs --out-dir=some/dir        -> writes under some/dir instead
//   SCENES_DIR=dev/stubs node build.mjs ...  -> pulls scenes from another folder

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const SCENE_NAMES = ['gargantua', 'lenia', 'mandelbulb'];

function parseArgs(argv) {
  let outDir = ROOT;
  for (const arg of argv) {
    if (arg.startsWith('--out-dir=')) {
      outDir = path.resolve(ROOT, arg.slice('--out-dir='.length));
    }
  }
  return { outDir };
}

function readFileOrDie(filePath, label) {
  try {
    return fs.readFileSync(filePath, 'utf8');
  } catch (err) {
    console.error(`[build] не удалось прочитать ${label}: ${filePath}`);
    console.error(`[build] ${err.message}`);
    process.exit(1);
  }
}

// Guards against an embedded "</script" (e.g. inside a string literal or
// comment in scene code) prematurely closing the inline <script> tag.
function escapeInlineScript(js) {
  return js.replace(/<\/(script)/gi, '<\\/$1');
}

function scriptTag(js, label) {
  return `<script>\n/* ${label} */\n${escapeInlineScript(js)}\n</script>`;
}

function main() {
  const { outDir } = parseArgs(process.argv.slice(2));

  const templatePath = path.join(ROOT, 'src/shell/template.html');
  const cssPath = path.join(ROOT, 'src/shell/shell.css');
  const shellJsPath = path.join(ROOT, 'src/shell/shell.js');
  const scenesDir = path.resolve(ROOT, process.env.SCENES_DIR || 'src/scenes');

  const template = readFileOrDie(templatePath, 'шаблон оболочки (template.html)');
  const css = readFileOrDie(cssPath, 'стили оболочки (shell.css)');
  const shellJs = readFileOrDie(shellJsPath, 'скрипт оболочки (shell.js)');

  const sceneScripts = SCENE_NAMES.map((name) => {
    const file = path.join(scenesDir, `${name}.js`);
    if (!fs.existsSync(file)) {
      console.error(`[build] отсутствует файл сцены: ${file}`);
      console.error(
        `[build] ожидались файлы ${SCENE_NAMES.map((n) => n + '.js').join(', ')} в ${scenesDir}` +
          ' (задайте SCENES_DIR, чтобы указать другую папку)'
      );
      process.exit(1);
    }
    return { name, code: readFileOrDie(file, `сцена «${name}»`) };
  });

  if (!template.includes('{{STYLE}}') || !template.includes('{{SCRIPTS}}')) {
    console.error('[build] в template.html отсутствуют плейсхолдеры {{STYLE}} и/или {{SCRIPTS}}');
    process.exit(1);
  }
  if (!template.includes('<!--HEAD_END-->')) {
    console.error('[build] в template.html отсутствует маркер <!--HEAD_END-->, разделяющий head/body');
    process.exit(1);
  }

  const scriptsHtml = sceneScripts
    .map(({ name, code }) => scriptTag(code, `сцена: ${name}`))
    .concat([scriptTag(shellJs, 'оболочка: shell.js')])
    .join('\n');

  // {{STYLE}} sits before the HEAD_END marker (fonts link + <style>), the
  // rest of the template (after the marker) is body markup + {{SCRIPTS}}.
  const withStyle = template.replace('{{STYLE}}', () => css);
  const [headPartRaw, bodyPartRaw] = withStyle.split('<!--HEAD_END-->');
  if (bodyPartRaw === undefined) {
    console.error('[build] не удалось разделить template.html по маркеру <!--HEAD_END-->');
    process.exit(1);
  }
  const headPart = headPartRaw.trim();
  const bodyPart = bodyPartRaw.replace('{{SCRIPTS}}', () => scriptsHtml).trim();

  const description =
    'Три невозможные вещи — «Гаргантюа», «Ления» и «Мандельбульб»: три живые GPU-сцены, ' +
    'вычисленные из математики в реальном времени, без единого изображения и без библиотек.';

  const indexHtml =
    '<!doctype html>\n' +
    '<html lang="ru">\n' +
    '<head>\n' +
    '<meta charset="UTF-8">\n' +
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n' +
    '<title>Три невозможные вещи</title>\n' +
    `<meta name="description" content="${description}">\n` +
    headPart +
    '\n</head>\n' +
    '<body>\n' +
    bodyPart +
    '\n</body>\n' +
    '</html>\n';

  // Fragment for claude.ai Artifacts: the host supplies its own doctype,
  // html/head/body and a skeleton (:root safe-area padding, off-white body
  // background) — our CSS below overrides body background and uses
  // position:fixed for the stage so that skeleton never shows through.
  const artifactHtml = '<title>Три невозможные вещи</title>\n' + headPart + '\n' + bodyPart + '\n';

  const indexPath = path.join(outDir, 'index.html');
  const artifactPath = path.join(outDir, 'dist/artifact.html');

  fs.mkdirSync(path.dirname(indexPath), { recursive: true });
  fs.mkdirSync(path.dirname(artifactPath), { recursive: true });
  fs.writeFileSync(indexPath, indexHtml, 'utf8');
  fs.writeFileSync(artifactPath, artifactHtml, 'utf8');

  const kb = (s) => (Buffer.byteLength(s, 'utf8') / 1024).toFixed(1);
  console.log(`[build] ${indexPath} — ${kb(indexHtml)} КБ`);
  console.log(`[build] ${artifactPath} — ${kb(artifactHtml)} КБ`);
}

main();
