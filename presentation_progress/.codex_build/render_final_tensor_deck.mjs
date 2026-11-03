import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const source = "C:/Projects/AIBSI/presentation_progress/output/NeuroTensor_Progress_Tensor_Hybrid_2026_Final.pptx";
const out = "C:/Projects/AIBSI/presentation_progress/.codex_build/tensor_progress/final_render";

async function writeBlob(target, blob) {
  await fs.writeFile(target, new Uint8Array(await blob.arrayBuffer()));
}

await fs.mkdir(out, { recursive: true });
const presentation = await PresentationFile.importPptx(await FileBlob.load(source));
for (const [index, slide] of presentation.slides.items.entries()) {
  await writeBlob(path.join(out, `slide-${String(index + 1).padStart(2, "0")}.png`), await presentation.export({ slide, format: "png", scale: 1 }));
}
await writeBlob(path.join(out, "montage.webp"), await presentation.export({ format: "webp", montage: true, scale: 0.5 }));
console.log(JSON.stringify({ slides: presentation.slides.items.length, output: out }));
