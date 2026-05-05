// Renders a Claude-generated research note with inline [n] citations turned
// into clickable chips that scroll to / highlight the matching source row.
"use client";
import { useMemo } from "react";

interface Props {
  text: string;
  onCiteClick?: (n: number) => void;
}

// Light-weight markdown-ish renderer: bullets, numbered lists, paragraphs,
// bold (**...**), markdown tables, and inline citations like [1] or [2,3].
export default function ResearchNote({ text, onCiteClick }: Props) {
  const blocks = useMemo(() => splitBlocks(text), [text]);

  return (
    <div className="research-note">
      {blocks.map((b, i) => {
        if (b.kind === "ul") {
          return (
            <ul key={i}>
              {b.items.map((it, j) => (
                <li key={j}>{renderInline(it, onCiteClick)}</li>
              ))}
            </ul>
          );
        }
        if (b.kind === "ol") {
          return (
            <ol key={i}>
              {b.items.map((it, j) => (
                <li key={j}>{renderInline(it, onCiteClick)}</li>
              ))}
            </ol>
          );
        }
        if (b.kind === "h") {
          const Tag = (b.level === 2 ? "h2" : "h3") as "h2" | "h3";
          return <Tag key={i}>{renderInline(b.text, onCiteClick)}</Tag>;
        }
        if (b.kind === "table") {
          return (
            <div key={i} style={{ overflowX: "auto", marginBottom: "12px" }}>
              <table className="memo-table">
                <thead>
                  <tr>
                    {b.headers.map((h, j) => (
                      <th key={j} style={colAlign(h, b.alignments[j])}>
                        {renderInline(h, onCiteClick)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {b.rows.map((row, j) => (
                    <tr key={j}>
                      {row.map((cell, k) => (
                        <td
                          key={k}
                          className="tnum"
                          style={colAlign(cell, b.alignments[k])}
                        >
                          {renderInline(cell, onCiteClick)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        }
        return <p key={i}>{renderInline(b.text, onCiteClick)}</p>;
      })}
    </div>
  );
}

// Returns inline style for a table cell based on the column alignment hint.
// Numeric-looking content defaults to right-align; plain text stays left.
function colAlign(
  content: string,
  alignment: "left" | "right" | "center"
): React.CSSProperties {
  if (alignment === "right") return { textAlign: "right" };
  if (alignment === "center") return { textAlign: "center" };
  // Auto-detect: if the cell content looks numeric, right-align it.
  const stripped = content.replace(/[$%,SGD\s]/g, "").replace(/\[.*?\]/g, "").trim();
  if (stripped !== "" && !isNaN(Number(stripped))) return { textAlign: "right" };
  return { textAlign: "left" };
}

type Block =
  | { kind: "p"; text: string }
  | { kind: "h"; level: 2 | 3; text: string }
  | { kind: "ul"; items: string[] }
  | { kind: "ol"; items: string[] }
  | { kind: "table"; headers: string[]; rows: string[][]; alignments: ("left" | "right" | "center")[] };

// Parse a pipe-delimited markdown table row into cell strings.
function parseTableRow(line: string): string[] {
  return line
    .replace(/^\|/, "")   // strip leading |
    .replace(/\|$/, "")   // strip trailing |
    .split("|")
    .map(c => c.trim());
}

// Detect alignment from a separator row cell like `---`, `:---`, `---:`, `:---:`.
function parseSepAlignment(cell: string): "left" | "right" | "center" {
  const c = cell.trim();
  const left = c.startsWith(":");
  const right = c.endsWith(":");
  if (left && right) return "center";
  if (right) return "right";
  return "left";
}

// Returns true if a table row is a separator row (all cells are dashes/colons).
function isSeparatorRow(cells: string[]): boolean {
  return cells.length > 0 && cells.every(c => /^:?-+:?$/.test(c.trim()));
}

function splitBlocks(text: string): Block[] {
  const lines = text.split("\n");
  const out: Block[] = [];
  let buf: string[] = [];
  let listItems: string[] = [];
  let listKind: "ul" | "ol" | null = null;
  let tableLines: string[] = [];

  const flushPara = () => {
    if (buf.length) {
      out.push({ kind: "p", text: buf.join(" ") });
      buf = [];
    }
  };
  const flushList = () => {
    if (listItems.length) {
      out.push({ kind: listKind!, items: listItems });
      listItems = [];
      listKind = null;
    }
  };
  const flushTable = () => {
    if (!tableLines.length) return;
    const allRows = tableLines.map(parseTableRow);
    // Find the separator row index
    const sepIdx = allRows.findIndex(isSeparatorRow);
    if (sepIdx === -1 || sepIdx === 0) {
      // Not a valid table — fall back to treating lines as paragraphs
      buf.push(...tableLines);
      tableLines = [];
      return;
    }
    const headers = allRows[sepIdx - 1];
    const alignments = allRows[sepIdx].map(parseSepAlignment);
    const rows = allRows.slice(sepIdx + 1).filter(r => r.length > 0);
    if (rows.length === 0) {
      buf.push(...tableLines);
      tableLines = [];
      return;
    }
    out.push({ kind: "table", headers, rows, alignments });
    tableLines = [];
  };

  for (const raw of lines) {
    const line = raw.trim();

    // Table line: starts with a pipe character
    if (line.startsWith("|")) {
      flushPara();
      flushList();
      tableLines.push(line);
      continue;
    }

    // Any non-pipe line terminates a table block in progress
    if (tableLines.length > 0) {
      flushTable();
    }

    if (!line) {
      flushPara();
      flushList();
      continue;
    }
    const h2 = line.match(/^##\s+(.*)$/);
    const h3 = line.match(/^###\s+(.*)$/);
    if (h2) {
      flushPara(); flushList();
      out.push({ kind: "h", level: 2, text: h2[1] });
      continue;
    }
    if (h3) {
      flushPara(); flushList();
      out.push({ kind: "h", level: 3, text: h3[1] });
      continue;
    }
    const ulMatch = line.match(/^[-*•]\s+(.*)$/);
    const olMatch = line.match(/^\d+\.\s+(.*)$/);
    if (ulMatch) {
      flushPara();
      if (listKind === "ol") flushList();
      listKind = "ul";
      listItems.push(ulMatch[1]);
      continue;
    }
    if (olMatch) {
      flushPara();
      if (listKind === "ul") flushList();
      listKind = "ol";
      listItems.push(olMatch[1]);
      continue;
    }
    flushList();
    buf.push(line);
  }
  // Flush anything remaining
  if (tableLines.length > 0) flushTable();
  flushPara();
  flushList();
  return out;
}

// Render bold + citation chips. Citations like [1], [2,3], [1, 2, 3].
function renderInline(text: string, onCite?: (n: number) => void) {
  const out: (string | JSX.Element)[] = [];
  // Split by bold first
  const boldPieces = text.split(/(\*\*[^*]+\*\*)/g);
  let key = 0;
  for (const piece of boldPieces) {
    const isBold = piece.startsWith("**") && piece.endsWith("**");
    const inner = isBold ? piece.slice(2, -2) : piece;
    const chunks = splitCitations(inner, onCite, () => key++);
    if (isBold) {
      out.push(<strong key={`b${key++}`}>{chunks}</strong>);
    } else {
      out.push(...chunks);
    }
  }
  return out;
}

function splitCitations(
  text: string,
  onCite: ((n: number) => void) | undefined,
  nextKey: () => number
): (string | JSX.Element)[] {
  // Match [1], [2,3], [1, 2, 3]
  const re = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
  const out: (string | JSX.Element)[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const nums = m[1].split(",").map(s => Number(s.trim())).filter(Number.isFinite);
    nums.forEach((n, i) => {
      out.push(
        <button
          key={`c${nextKey()}`}
          type="button"
          className="cite"
          onClick={() => onCite?.(n)}
          title={`Source ${n}`}
        >
          {n}
        </button>
      );
      if (i < nums.length - 1) out.push(", ");
    });
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}
