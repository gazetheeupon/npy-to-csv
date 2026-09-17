import { parseNpyHeader, buildTable, isNpyMagic } from "./npy-parser.js";

const PREVIEW_ROWS = 50;
const PREVIEW_COLS = 20;
const MAX_CSV_CELLS = 8_000_000;

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const fnameEl = document.getElementById("fname");
const statusEl = document.getElementById("status");
const varsCard = document.getElementById("varsCard");
const varsHeading = document.getElementById("varsHeading");
const varsBody = document.getElementById("varsBody");
const previewCard = document.getElementById("previewCard");
const previewHeading = document.getElementById("previewHeading");
const previewTable = document.getElementById("previewTable");
const truncNote = document.getElementById("truncNote");
const exportBtn = document.getElementById("exportBtn");
const exportStatusEl = document.getElementById("exportStatus");

let variables = []; // [{name, buffer, header?, table?, error?}]
let selectedIndex = -1;

function setStatus(msg, isError) {
  statusEl.textContent = msg || "";
  statusEl.className = isError ? "error" : "";
}

function setExportStatus(msg, isWarn) {
  exportStatusEl.textContent = msg || "";
  exportStatusEl.className = isWarn ? "warn" : "";
}

function resetUI() {
  varsCard.style.display = "none";
  previewCard.style.display = "none";
  varsBody.innerHTML = "";
  previewTable.innerHTML = "";
  truncNote.textContent = "";
  setExportStatus("");
  variables = [];
  selectedIndex = -1;
}

async function handleFile(file) {
  resetUI();
  fnameEl.textContent = file.name;
  setStatus("Reading file...");
  let buffer;
  try {
    buffer = await file.arrayBuffer();
  } catch (e) {
    setStatus(`Could not read file: ${e.message}`, true);
    return;
  }

  const bytes = new Uint8Array(buffer);
  const isZip = bytes.length >= 4 && bytes[0] === 0x50 && bytes[1] === 0x4b; // "PK"

  try {
    if (isZip) {
      loadNpz(bytes, file.name);
    } else if (isNpyMagic(bytes)) {
      loadSingleNpy(buffer, file.name);
    } else {
      throw new Error("This doesn't look like a .npy file or a .npz (ZIP) archive.");
    }
  } catch (e) {
    setStatus(e.message, true);
    return;
  }

  if (variables.length === 0) {
    setStatus("No readable NumPy arrays were found in this file.", true);
    return;
  }

  setStatus("");
  renderVarsTable();
  varsCard.style.display = "";
  varsHeading.textContent = variables.length > 1 ? `Arrays (${variables.length})` : "Array";
  selectVariable(0);
}

function loadSingleNpy(buffer, fileName) {
  const name = fileName.replace(/\.npy$/i, "");
  const entry = { name, buffer };
  try {
    entry.header = parseNpyHeader(buffer);
  } catch (e) {
    entry.error = e.message;
  }
  variables = [entry];
}

function loadNpz(bytes, fileName) {
  if (typeof window.fflateUnzipSync !== "function") {
    throw new Error("The .npz unzip helper failed to load. Try reloading the page.");
  }
  let entries;
  try {
    entries = window.fflateUnzipSync(bytes);
  } catch (e) {
    throw new Error(`Could not read this .npz archive as a ZIP file: ${e.message}`);
  }
  const names = Object.keys(entries).filter((n) => n.toLowerCase().endsWith(".npy"));
  if (names.length === 0) {
    throw new Error("This .npz archive doesn't contain any .npy arrays.");
  }
  names.sort();
  variables = names.map((n) => {
    const raw = entries[n];
    // Copy into a fresh, zero-offset ArrayBuffer so byte offsets in the parser are simple.
    const buffer = raw.slice().buffer;
    const displayName = n.replace(/\.npy$/i, "");
    const entry = { name: displayName, buffer };
    try {
      entry.header = parseNpyHeader(buffer);
    } catch (e) {
      entry.error = e.message;
    }
    return entry;
  });
}

function renderVarsTable() {
  varsBody.innerHTML = "";
  variables.forEach((v, i) => {
    const tr = document.createElement("tr");
    tr.className = "var-row" + (v.error ? " errored" : "");
    if (v.error) {
      tr.innerHTML = `<td>${escapeHtml(v.name)}</td><td colspan="4">${escapeHtml(v.error)}</td>`;
    } else {
      const h = v.header;
      tr.innerHTML = `<td>${escapeHtml(v.name)}</td><td>${escapeHtml(h.dtypeLabel)}</td>` +
        `<td>(${h.shape.join(", ")})</td><td>${h.fortranOrder ? "Fortran" : "C"}</td>` +
        `<td>${h.numElements.toLocaleString()}</td>`;
      tr.addEventListener("click", () => selectVariable(i));
    }
    varsBody.appendChild(tr);
  });
}

function selectVariable(i) {
  const v = variables[i];
  if (!v || v.error) return;
  selectedIndex = i;
  [...varsBody.children].forEach((tr, idx) => tr.classList.toggle("selected", idx === i));

  const h = v.header;
  let table;
  try {
    table = buildTable(v.buffer, h);
  } catch (e) {
    previewCard.style.display = "";
    previewHeading.textContent = `Preview: ${v.name}`;
    previewTable.innerHTML = "";
    truncNote.textContent = "";
    setExportStatus(e.message, true);
    exportBtn.disabled = true;
    return;
  }
  v.table = table;

  previewCard.style.display = "";
  previewHeading.textContent = `Preview: ${v.name} — dtype ${h.dtypeLabel}, shape (${h.shape.join(", ")})`;
  renderPreview(table);
  setExportStatus("");
  exportBtn.disabled = false;
}

function renderPreview(table) {
  const rShown = Math.min(table.rows, PREVIEW_ROWS);
  const cShown = Math.min(table.cols, PREVIEW_COLS);

  let html = "<thead><tr><th></th>";
  for (let c = 0; c < cShown; c++) html += `<th>${escapeHtml(table.colHeaders[c])}</th>`;
  html += "</tr></thead><tbody>";
  for (let r = 0; r < rShown; r++) {
    html += `<tr><th>${r}</th>`;
    for (let c = 0; c < cShown; c++) {
      html += `<td>${escapeHtml(formatCell(table.getCell(r, c)))}</td>`;
    }
    html += "</tr>";
  }
  html += "</tbody>";
  previewTable.innerHTML = html;

  const notes = [];
  if (table.rows > rShown) notes.push(`showing first ${rShown} of ${table.rows} rows`);
  if (table.cols > cShown) notes.push(`first ${cShown} of ${table.cols} columns`);
  truncNote.textContent = notes.length ? `(${notes.join(", ")} — full data is included in the CSV export)` : "";
}

function formatCell(v) {
  if (typeof v === "boolean") return v ? "True" : "False";
  if (typeof v === "number") {
    if (Number.isInteger(v)) return String(v);
    return String(v);
  }
  return String(v);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function csvEscape(v) {
  const s = formatCell(v);
  if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

exportBtn.addEventListener("click", () => {
  const v = variables[selectedIndex];
  if (!v || !v.table) return;
  const { rows, cols, colHeaders, getCell } = v.table;
  const totalCells = rows * cols;

  if (totalCells > MAX_CSV_CELLS) {
    setExportStatus(
      `This array has ${totalCells.toLocaleString()} cells, above the ${MAX_CSV_CELLS.toLocaleString()}-cell export limit. ` +
      `Try a smaller array, or select a different variable.`,
      true
    );
    return;
  }
  if (totalCells > 1_000_000) {
    const ok = window.confirm(
      `This will export ${totalCells.toLocaleString()} cells (${rows.toLocaleString()} rows x ${cols.toLocaleString()} cols). ` +
      `That may take a moment and produce a large file. Continue?`
    );
    if (!ok) return;
  }

  setExportStatus("Building CSV...");
  setTimeout(() => {
    const lines = [];
    lines.push(["row", ...colHeaders].map(csvEscape).join(","));
    for (let r = 0; r < rows; r++) {
      const row = [r];
      for (let c = 0; c < cols; c++) row.push(getCell(r, c));
      lines.push(row.map(csvEscape).join(","));
    }
    const csv = lines.join("\r\n") + "\r\n";
    const blob = new Blob([csv], { type: "text/csv" });
    downloadBlob(blob, `${v.name.replace(/[^a-z0-9_.-]+/gi, "_")}.csv`);
    setExportStatus(`Exported ${rows.toLocaleString()} rows x ${cols.toLocaleString()} cols.`);
  }, 10);
});

// ---- file input wiring ----
dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});
["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file) handleFile(file);
});
