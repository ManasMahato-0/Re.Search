import { useState } from "react";
import { ArrowUpRight, Search, Sparkles } from "lucide-react";
import heroArt from "@/assets/hero.jpg";

const SUGGESTIONS = [
  "origins of paper",
  "how tides shape coastlines",
  "physics of a violin string",
];

export default function App() {
  const [q, setQ] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);

  const handleSearch = async (e) => {
    e.preventDefault(); 
    if (!q.trim()) return;

    setLoading(true);
    setHasSearched(true);
    
    try {
      // Same-origin serverless proxy (frontend/api/search.js) — holds the
      // HF token server-side and talks to the Space's Gradio API for us.
      // Falls back to a local FastAPI backend during development.
      const API_URL = import.meta.env.VITE_API_URL || "";
      const base = API_URL || "";
      const response = await fetch(
        base
          ? `${base}/search?q=${encodeURIComponent(q)}`
          : `/api/search?q=${encodeURIComponent(q)}`
      );
      const data = await response.json();
      if (data.status === 'success') {
        setResults(data.results);
      } else {
        console.error("Search failed:", data.message || data);
        setResults([]);
      }
    } catch (error) {
      console.error("Search failed:", error);
    } finally {
      setLoading(false);
    }
  };

  const resetSearch = (e) => {
    e.preventDefault();
    setHasSearched(false);
    setQ("");
    setResults([]);
  };

  return (
    <div className="min-h-screen bg-background text-foreground paper-grain selection:bg-[color:var(--moss)] selection:text-white">
      
      {/* Nav (Always visible) */}
      <header className="mx-auto flex max-w-7xl items-center justify-between px-6 py-6">
        <a href="/" onClick={resetSearch} className="flex items-baseline gap-1 font-serif text-2xl">
          <span className="italic">Re</span>
          <span>Search</span>
          <span className="ml-1 h-1.5 w-1.5 rounded-full bg-[color:var(--moss)]" />
        </a>

      </header>

      {/* CONDITIONAL RENDERING: Show Homepage or Results */}
      {!hasSearched ? (
        
        <div className="animate-in fade-in duration-700">
          {/* Hero */}
          <section className="mx-auto grid max-w-7xl grid-cols-1 gap-12 px-6 pt-12 pb-24 lg:grid-cols-[1.15fr_1fr] lg:gap-16 lg:pt-20">
            <div className="flex flex-col justify-center">
              <div className="mb-8 flex items-center gap-3 font-mono text-xs uppercase tracking-[0.2em] text-muted-foreground">
                <span className="h-px w-8 bg-foreground/40" />
                Vol. I · Summer 2026
              </div>

              <h1 className="font-serif text-6xl leading-[0.95] tracking-tight sm:text-7xl lg:text-[5.5rem]">
                For the <em className="text-moss">curious</em>,
                <br />
                not the{" "}
                <span className="italic text-muted-foreground">consumer</span>.
              </h1>

              <p className="mt-8 max-w-md text-lg leading-relaxed text-muted-foreground">
                Sources, not slop. Ask a real question.
              </p>

              {/* Functional Search Form */}
              <form onSubmit={handleSearch} className="mt-10 max-w-2xl">
                <div className="group flex items-center gap-3 rounded-full border border-foreground/25 bg-card px-5 py-3 shadow-[0_1px_0_0_oklch(0.18_0.02_60/0.08),0_20px_40px_-24px_oklch(0.18_0.02_60/0.25)] transition focus-within:border-moss focus-within:shadow-[0_0_0_4px_oklch(0.42_0.07_145/0.12)]">
                  <Search className="h-5 w-5 text-muted-foreground" strokeWidth={1.5} />

                  <input
                    value={q}
                    onChange={(e) => setQ(e.target.value)}
                    placeholder="Query the record…"
                    className="flex-1 bg-transparent font-serif text-xl italic placeholder:text-muted-foreground/60 focus:outline-none"
                  />

                  <button
                    type="submit"
                    className="ml-1 inline-flex items-center gap-1.5 rounded-full bg-foreground px-4 py-1.5 text-sm text-background transition hover:bg-moss"
                    aria-label="Search"
                  >
                    Enquire
                  </button>
                </div>

                {/* Suggestion Pills */}
                <div className="mt-5 flex flex-wrap items-center gap-2">
                  {SUGGESTIONS.map((s) => (
                    <button
                      key={s}
                      type="button"
                      onClick={() => setQ(s)}
                      className="rounded-full border border-foreground/15 px-3 py-1 text-sm text-foreground/80 transition hover:border-foreground/50 hover:bg-foreground/5"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </form>
            </div>

            {/* Hero Image */}
            <figure className="relative">
              <div className="absolute inset-0 -rotate-1 rounded-sm bg-(--ochre)/10" />

              <div className="relative overflow-hidden rounded-sm border border-foreground/15 bg-card">
                <img
                  src={heroArt}
                  alt="Ink illustration"
                  className="h-full w-full object-cover mix-blend-multiply"
                />

                <figcaption className="flex items-center justify-between border-t border-foreground/15 bg-card px-4 py-3 font-mono text-[11px] uppercase tracking-widest text-muted-foreground">
                  <span>Plate 01</span>
                  <span>Ink on cream</span>
                </figcaption>
              </div>

              <div className="absolute -bottom-8 -left-6 hidden max-w-60 -rotate-2 border border-foreground/15 bg-background p-4 shadow-xl md:block">
                <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-widest text-moss">
                  <Sparkles className="h-3 w-3" />
                  Footnote
                </div>

                <p className="mt-2 font-serif text-base italic leading-snug">
                  “Search well by asking well.”
                </p>
              </div>
            </figure>
          </section>

          {/* Method */}
          <section
            id="method"
            className="border-y border-foreground/15 bg-(--paper)/60"
          >
            <div className="mx-auto grid max-w-7xl grid-cols-1 gap-0 md:grid-cols-3">
              {[
                { n: "I.", t: "Cite the record" },
                { n: "II.", t: "Weigh the evidence" },
                { n: "III.", t: "Read at length" },
              ].map((c, i) => (
                <div
                  key={c.n}
                  className={`p-10 lg:p-14 ${
                    i < 2 ? "md:border-r border-foreground/15" : ""
                  }`}
                >
                  <div className="flex items-baseline gap-4">
                    <span className="font-serif text-5xl italic text-ochre">
                      {c.n}
                    </span>
                    <h3 className="font-serif text-2xl">{c.t}</h3>
                  </div>
                </div>
              ))}
            </div>
          </section>
        </div>

      ) : (

        /* --- THE SEARCH RESULTS STATE --- */
        <main className="mx-auto max-w-4xl px-6 pt-12 pb-24 animate-in fade-in slide-in-from-bottom-4 duration-500">
          
          {/* Top Search Bar */}
          <form onSubmit={handleSearch} className="mb-16">
            <div className="group flex items-center gap-3 rounded-full border border-foreground/25 bg-card px-5 py-3 shadow-sm transition focus-within:border-moss">
              <Search className="h-5 w-5 text-muted-foreground" strokeWidth={1.5} />
              <input
                value={q}
                onChange={(e) => setQ(e.target.value)}
                className="flex-1 bg-transparent font-serif text-xl italic placeholder:text-muted-foreground/60 focus:outline-none"
              />
              <button type="submit" className="ml-1 inline-flex items-center gap-1.5 rounded-full bg-foreground px-4 py-1.5 text-sm text-background transition hover:bg-moss disabled:opacity-50">
                {loading ? '...' : 'Enquire'}
              </button>
            </div>
          </form>

          {/* Results List */}
          {loading ? (
             <div className="text-center font-serif text-2xl italic text-muted-foreground mt-20">
               Consulting the archives...
             </div>
          ) : (
             <div className="space-y-12">
               {results.map((item, index) => (
                 <article key={index} className="border-b border-foreground/10 pb-10">
                   <div className="flex gap-4 items-baseline mb-3">
                     <span className="font-serif text-ochre italic text-xl">{index + 1}.</span>
                     <a href={item.url} target="_blank" rel="noreferrer" className="group">
                       <h3 className="font-serif text-2xl font-medium text-foreground group-hover:text-[color:var(--moss)] transition-colors leading-snug">
                         {item.title}
                       </h3>
                     </a>
                   </div>
                   
                   <span className="block text-xs font-mono uppercase tracking-widest text-muted-foreground ml-8 mb-4 break-all">
                     {item.url}
                   </span>
                   
                   <p className="ml-8 text-lg text-foreground/80 leading-relaxed max-w-2xl">
                     {item.snippet}
                   </p>
                 </article>
               ))}

               {results.length === 0 && !loading && (
                 <div className="text-center font-serif text-xl italic text-muted-foreground mt-20">
                    No records found for this query.
                 </div>
               )}
             </div>
          )}
        </main>
      )}
    </div>
  );
}