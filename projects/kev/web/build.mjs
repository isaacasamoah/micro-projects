import { execFileSync } from "node:child_process";
import { cp, mkdir, readFile, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const web = dirname(fileURLToPath(import.meta.url));
const kev = resolve(web, "..");
const rustBin = join(
  process.env.CARGO_HOME ?? join(homedir(), ".cargo"),
  "bin",
);
const run = (command, args) =>
  execFileSync(command, args, { cwd: web, stdio: "inherit" });
run(join(rustBin, "cargo"), [
  "build",
  "--locked",
  "--target-dir",
  join(kev, "target"),
  "--manifest-path",
  join(kev, "Cargo.toml"),
  "--target",
  "wasm32-unknown-unknown",
  "--features",
  "browser",
  "--release",
]);
run(join(rustBin, "wasm-bindgen"), [
  join(kev, "target/wasm32-unknown-unknown/release/kev.wasm"),
  "--target",
  "web",
  "--out-dir",
  join(web, "generated"),
]);
run(process.execPath, [join(web, "node_modules/typescript/bin/tsc")]);
await mkdir(join(web, "dist"), { recursive: true });
await cp(join(web, "public"), join(web, "dist"), { recursive: true });
await cp(join(kev, "data"), join(web, "dist/data"), { recursive: true });
await cp(join(web, "generated"), join(web, "dist/generated"), {
  recursive: true,
});

const escape = (text) =>
  text.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
let walkthrough = await readFile(join(web, "public/walkthrough.html"), "utf8");
for (const name of ["lib", "attention", "math", "browser"]) {
  const source = await readFile(join(kev, `src/${name}.rs`), "utf8");
  walkthrough = walkthrough.replace(
    `<!-- source:${name} -->`,
    `<pre><code>${escape(source)}</code></pre>`,
  );
}
await writeFile(join(web, "dist/walkthrough.html"), walkthrough);
console.log("Built Rust + TypeScript visual in web/dist. Run npm run serve.");
