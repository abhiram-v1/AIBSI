import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const root = "C:/Projects/AIBSI";
const source = path.join(root, "presentation_progress/output/NeuroTensor_Status_Presentation_10_Slides_EEG_Pipeline.pptx");
const out = path.join(root, "presentation_progress/.codex_build/template_render");

async function writeBlob(target, blob) {
  await fs.writeFile(target, new Uint8Array(await blob.arrayBuffer()));
}

await fs.mkdir(out, { recursive: true });
const presentation = await PresentationFile.importPptx(await FileBlob.load(source));
const inspected = await presentation.inspect({
  select: "$.slides[*]",
  include: ["id", "index", "layout", "title", "elements.id", "elements.kind", "elements.text", "elements.position", "elements.style"],
});
await fs.writeFile(path.join(out, "content.ndjson"), inspected.ndjson, "utf8");
for (const [index, slide] of presentation.slides.items.entries()) {
  await writeBlob(path.join(out, `slide-${String(index + 1).padStart(2, "0")}.png`), await presentation.export({ slide, format: "png", scale: 1 }));
}
await writeBlob(path.join(out, "montage.webp"), await presentation.export({ format: "webp", montage: true, scale: 0.5 }));
console.log(JSON.stringify({ slideCount: presentation.slides.items.length, slideSize: presentation.slideSize, masters: presentation.masters.items.length, layouts: presentation.layouts.items.length }, null, 2));
