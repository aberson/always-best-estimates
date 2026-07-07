export default function App() {
  return (
    <main
      style={{
        fontFamily: "system-ui, sans-serif",
        maxWidth: 720,
        margin: "3rem auto",
        padding: "0 1rem",
      }}
    >
      <h1>always-best-estimates</h1>
      <section
        style={{
          border: "1px solid #ccc",
          borderRadius: 8,
          padding: "1rem 1.25rem",
          marginTop: "1rem",
        }}
      >
        <h2 style={{ marginTop: 0 }}>Pipeline stages</h2>
        <p>Stage cards (ingest, features, forecast, blend, optimize) arrive in Step 10.</p>
      </section>
    </main>
  );
}
