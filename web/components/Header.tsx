// Top page header — mirrors 001 layout. Brand stripe + product mark + as-of date.
export default function Header() {
  const today = new Date();
  const stamp = today
    .toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" })
    .toUpperCase();
  return (
    <header className="bg-white border-b border-rule">
      <div className="memo-header-rule" />
      <div className="max-w-[1400px] mx-auto px-8 py-5 flex items-end justify-between">
        <div>
          <div className="font-mono text-[10.5px] tracking-memo uppercase text-turf-500">
            Orikai · SEA Buyside Research Tooling
          </div>
          <h1 className="font-serif text-3xl text-ink mt-1 leading-tight">
            Keppel DC REIT — Deep-Dive Research Assistant
          </h1>
          <div className="text-[13px] text-graphite mt-1">
            Annual reports · Quarterly disclosures · Broker notes · SGX filings · Live AJBU.SI price · Cited research-note output
          </div>
        </div>
        <div className="text-right font-mono text-[11px] text-graphite">
          <div className="tracking-memo uppercase">As of</div>
          <div className="tnum mt-1">{stamp}</div>
        </div>
      </div>
    </header>
  );
}
