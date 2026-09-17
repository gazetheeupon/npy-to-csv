// npy-parser.js — dependency-free parser for NumPy .npy array files.
// Format reference: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
//
// A .npy file is:
//   bytes 0-5   magic string \x93NUMPY
//   byte  6     major version
//   byte  7     minor version
//   v1.0: bytes 8-9   header length (uint16 LE)      header starts at byte 10
//   v2/3: bytes 8-11  header length (uint32 LE)       header starts at byte 12
//   header        ASCII, a Python dict literal: {'descr': '<f8', 'fortran_order': False, 'shape': (3, 4), }
//                 padded with spaces and a trailing \n to a 64-byte-aligned total.
//   data          raw array bytes, in the layout described by descr/fortran_order/shape.

const MAGIC = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]; // \x93NUMPY

export function isNpyMagic(bytes) {
  if (bytes.length < 6) return false;
  for (let i = 0; i < 6; i++) if (bytes[i] !== MAGIC[i]) return false;
  return true;
}

function parseDescr(descr) {
  // descr like '<f8', '>i4', '|b1', '|S10', '<U21', '<c16'
  const m = /^([<>=|])([a-zA-Z])(\d*)$/.exec(descr.trim());
  if (!m) {
    throw new Error(`Unrecognized dtype descriptor "${descr}". Structured/record dtypes are not supported.`);
  }
  const [, endian, typeChar, digits] = m;
  const count = digits ? parseInt(digits, 10) : 1;
  // For numeric/bool/complex dtypes the trailing number is the item width in BYTES.
  // For 'U' (unicode) it is a CHARACTER count, and each character is stored as a
  // 4-byte (UCS-4) code point, so the true item width in bytes is count * 4.
  // For 'S' (byte string) the count already is the byte width.
  const itemBytes = typeChar === "U" ? count * 4 : count;
  return { endian, typeChar, itemBytes, charCount: count, descr };
}

function dtypeLabel(dt) {
  const names = {
    f: "float", i: "int", u: "uint", b: "bool", c: "complex", S: "bytes", U: "unicode",
  };
  const base = names[dt.typeChar] || dt.typeChar;
  if (dt.typeChar === "S") return `bytes${dt.charCount}`;
  if (dt.typeChar === "U") return `unicode${dt.charCount}`;
  if (dt.typeChar === "b") return "bool";
  return `${base}${dt.itemBytes * 8}`;
}

export function parseNpyHeader(buffer) {
  const bytes = new Uint8Array(buffer);
  if (bytes.length < 10) throw new Error("File is too small to be a valid .npy file.");
  if (!isNpyMagic(bytes)) throw new Error('Missing NPY magic bytes ("\\x93NUMPY"). This does not look like a .npy file.');

  const major = bytes[6];
  const minor = bytes[7];
  const view = new DataView(buffer);
  let headerLen, headerStart;
  if (major === 1) {
    headerLen = view.getUint16(8, true);
    headerStart = 10;
  } else if (major === 2 || major === 3) {
    headerLen = view.getUint32(8, true);
    headerStart = 12;
  } else {
    throw new Error(`Unsupported .npy version ${major}.${minor}.`);
  }
  const dataOffset = headerStart + headerLen;
  if (bytes.length < dataOffset) throw new Error("File is truncated: header extends past end of file.");

  const headerStr = new TextDecoder("latin1").decode(bytes.subarray(headerStart, headerStart + headerLen));

  const descrMatch = /'descr'\s*:\s*'([^']*)'/.exec(headerStr);
  const fortranMatch = /'fortran_order'\s*:\s*(True|False)/.exec(headerStr);
  const shapeMatch = /'shape'\s*:\s*\(([^)]*)\)/.exec(headerStr);
  if (!descrMatch || !fortranMatch || !shapeMatch) {
    throw new Error("Could not parse the .npy header dictionary.");
  }

  const dtype = parseDescr(descrMatch[1]);
  const fortranOrder = fortranMatch[1] === "True";
  const shapeStr = shapeMatch[1].trim();
  const shape = shapeStr === "" ? [] : shapeStr.split(",").map((s) => s.trim()).filter((s) => s.length).map((s) => parseInt(s, 10));

  const numElements = shape.length === 0 ? 1 : shape.reduce((a, b) => a * b, 1);
  const expectedBytes = numElements * dtype.itemBytes;
  if (bytes.length < dataOffset + expectedBytes) {
    throw new Error(
      `File is truncated: expected ${expectedBytes} bytes of array data but only ${bytes.length - dataOffset} are present.`
    );
  }

  return {
    major, minor, dtype, fortranOrder, shape, numElements,
    dataOffset, dataByteLength: expectedBytes,
    dtypeLabel: dtypeLabel(dtype),
  };
}

function computeStrides(shape, fortranOrder) {
  const n = shape.length;
  const strides = new Array(n);
  if (fortranOrder) {
    let acc = 1;
    for (let k = 0; k < n; k++) { strides[k] = acc; acc *= shape[k]; }
  } else {
    let acc = 1;
    for (let k = n - 1; k >= 0; k--) { strides[k] = acc; acc *= shape[k]; }
  }
  return strides;
}

function decodeFloat16(uint16) {
  const sign = (uint16 & 0x8000) ? -1 : 1;
  const exponent = (uint16 >> 10) & 0x1f;
  const fraction = uint16 & 0x3ff;
  if (exponent === 0) return sign * Math.pow(2, -14) * (fraction / 1024);
  if (exponent === 0x1f) return fraction ? NaN : sign * Infinity;
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

// Reads a single scalar element at absolute byte offset `off` in `view`, per dtype.
function readScalar(view, off, dtype) {
  const little = dtype.endian === "<" || dtype.endian === "=" || dtype.endian === "|";
  switch (dtype.typeChar) {
    case "b":
      return view.getUint8(off) !== 0;
    case "i": {
      if (dtype.itemBytes === 1) return view.getInt8(off);
      if (dtype.itemBytes === 2) return view.getInt16(off, little);
      if (dtype.itemBytes === 4) return view.getInt32(off, little);
      if (dtype.itemBytes === 8) return Number(view.getBigInt64(off, little));
      break;
    }
    case "u": {
      if (dtype.itemBytes === 1) return view.getUint8(off);
      if (dtype.itemBytes === 2) return view.getUint16(off, little);
      if (dtype.itemBytes === 4) return view.getUint32(off, little);
      if (dtype.itemBytes === 8) return Number(view.getBigUint64(off, little));
      break;
    }
    case "f": {
      if (dtype.itemBytes === 2) return decodeFloat16(view.getUint16(off, little));
      if (dtype.itemBytes === 4) return view.getFloat32(off, little);
      if (dtype.itemBytes === 8) return view.getFloat64(off, little);
      break;
    }
    case "c": {
      const half = dtype.itemBytes / 2;
      let re, im;
      if (half === 4) { re = view.getFloat32(off, little); im = view.getFloat32(off + 4, little); }
      else { re = view.getFloat64(off, little); im = view.getFloat64(off + 8, little); }
      const sign = im < 0 ? "-" : "+";
      return `${re}${sign}${Math.abs(im)}j`;
    }
    case "S": {
      const bytes = new Uint8Array(view.buffer, view.byteOffset + off, dtype.itemBytes);
      let end = bytes.indexOf(0);
      if (end === -1) end = bytes.length;
      return new TextDecoder("utf-8", { fatal: false }).decode(bytes.subarray(0, end));
    }
    case "U": {
      const nChars = dtype.itemBytes / 4;
      let out = "";
      for (let i = 0; i < nChars; i++) {
        const cp = view.getUint32(off + i * 4, little);
        if (cp === 0) break;
        out += String.fromCodePoint(cp);
      }
      return out;
    }
    default:
      throw new Error(`Unsupported dtype "${dtype.descr}".`);
  }
  throw new Error(`Unsupported item width ${dtype.itemBytes} for dtype "${dtype.descr}".`);
}

// Generic row/col accessor: treats shape[0] as "rows" and flattens shape[1:] into
// "cols" in row-major (C) order, regardless of the array's actual storage order.
// Works for 0-D, 1-D, 2-D and N-D arrays.
export function buildTable(buffer, header) {
  const { shape, dtype, dataOffset, fortranOrder } = header;
  const view = new DataView(buffer);
  const strides = computeStrides(shape, fortranOrder);

  let rows, innerShape;
  if (shape.length === 0) {
    rows = 1;
    innerShape = [];
  } else {
    rows = shape[0];
    innerShape = shape.slice(1);
  }
  const cols = innerShape.length === 0 ? 1 : innerShape.reduce((a, b) => a * b, 1);

  // column headers, e.g. for innerShape [2,3] -> "0-0","0-1","0-2","1-0",...
  const colHeaders = [];
  if (innerShape.length === 0) {
    colHeaders.push("value");
  } else {
    const idx = new Array(innerShape.length).fill(0);
    for (let c = 0; c < cols; c++) {
      colHeaders.push(idx.join("-"));
      for (let d = innerShape.length - 1; d >= 0; d--) {
        idx[d]++;
        if (idx[d] < innerShape[d]) break;
        idx[d] = 0;
      }
    }
  }

  function getCell(r, c) {
    const idx = new Array(shape.length);
    if (shape.length === 0) return readScalar(view, dataOffset, dtype);
    idx[0] = r;
    if (innerShape.length) {
      let rem = c;
      for (let d = innerShape.length - 1; d >= 0; d--) {
        idx[1 + d] = rem % innerShape[d];
        rem = Math.floor(rem / innerShape[d]);
      }
    }
    let off = dataOffset;
    for (let k = 0; k < shape.length; k++) off += idx[k] * strides[k] * dtype.itemBytes;
    return readScalar(view, off, dtype);
  }

  return { rows, cols, colHeaders, getCell };
}
