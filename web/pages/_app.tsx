import "../styles/globals.css";
import type { AppProps } from "next/app";
import Head from "next/head";

export default function App({ Component, pageProps }: AppProps) {
  return (
    <>
      <Head>
        <title>Keppel DC REIT Research · Orikai</title>
        <meta
          name="description"
          content="Deep-dive Q&A over Keppel DC REIT (SGX: AJBU) annual reports, broker notes, SGX filings, and live price data. Research-note output with cited sources."
        />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link
          rel="icon"
          href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' fill='%2304724D'/%3E%3Ctext x='50%25' y='58%25' text-anchor='middle' font-family='Source Serif 4, Georgia, serif' font-size='20' fill='white'%3EO%3C/text%3E%3C/svg%3E"
        />
      </Head>
      <Component {...pageProps} />
    </>
  );
}
