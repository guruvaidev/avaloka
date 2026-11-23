// The Preview state of /instant: the upload's data profile, as the API
// reported it. Every number shown here was read from a response; the
// component formats and never measures.
import type { DataProfilePreview, InstantProfileStatus } from "@/lib/api/instant";

const fmtInt = (n: number | null) => (n == null ? "—" : n.toLocaleString());
const fmtPct = (n: number | null) => (n == null ? "—" : `${n}%`);

export function ProfilePreview({
  fileName,
  preview,
  profileStatus,
  profileError,
  issues,
}: {
  fileName: string;
  preview: DataProfilePreview;
  profileStatus: InstantProfileStatus;
  profileError: string | null;
  /** The upload's own issues plus the landed profile's concerns. */
  issues: string[];
}) {
  return (
    <section aria-label="Preview" className="flex flex-col gap-4" data-testid="instant-preview">
      <div className="flex flex-col gap-1">
        <h2 className="text-lg font-semibold text-primary">Data profile</h2>
        <p className="truncate text-sm text-tertiary" title={fileName}>
          {fileName}
        </p>
      </div>

      <dl className="grid grid-cols-2 gap-3">
        <div className="rounded-xl border border-secondary bg-secondary p-4">
          <dt className="text-xs text-tertiary">Rows</dt>
          <dd className="text-xl font-semibold text-primary" data-testid="instant-rows">
            {fmtInt(preview.rows)}
          </dd>
        </div>
        <div className="rounded-xl border border-secondary bg-secondary p-4">
          <dt className="text-xs text-tertiary">Columns</dt>
          <dd className="text-xl font-semibold text-primary" data-testid="instant-columns">
            {fmtInt(preview.columns)}
          </dd>
        </div>
      </dl>

      {preview.columnsTable.length ? (
        <div className="overflow-x-auto rounded-xl border border-secondary">
          <table
            className="w-full min-w-[320px] text-left text-sm"
            data-testid="instant-column-table"
          >
            <thead className="bg-secondary text-xs text-tertiary">
              <tr>
                <th scope="col" className="px-3 py-2 font-medium">
                  Column
                </th>
                <th scope="col" className="px-3 py-2 font-medium">
                  Type
                </th>
                <th scope="col" className="px-3 py-2 text-right font-medium">
                  Missing
                </th>
                <th scope="col" className="px-3 py-2 text-right font-medium">
                  Distinct
                </th>
              </tr>
            </thead>
            <tbody>
              {preview.columnsTable.map((c) => (
                <tr key={c.name} className="border-t border-secondary">
                  <td
                    className="max-w-[160px] truncate px-3 py-2 font-medium text-primary"
                    title={c.name}
                  >
                    {c.name}
                  </td>
                  <td className="px-3 py-2 text-secondary">{c.type}</td>
                  <td className="px-3 py-2 text-right tabular-nums text-secondary">
                    {fmtPct(c.missingPct)}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums text-secondary">
                    {fmtInt(c.distinct)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="text-sm text-tertiary">No column statistics came back with this upload.</p>
      )}

      <div className="flex flex-col gap-2" data-testid="instant-issues">
        <h3 className="text-sm font-semibold text-primary">Quality issues</h3>
        {issues.length ? (
          <ul className="list-disc space-y-1 pl-5 text-sm text-secondary">
            {issues.map((issue, i) => (
              <li key={i}>{issue}</li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-tertiary">
            {profileStatus === "pending"
              ? "Checking the data…"
              : profileStatus === "failed"
                ? (profileError ?? "The data profile could not be built.")
                : "None reported."}
          </p>
        )}
        {profileStatus === "pending" && issues.length ? (
          <p className="text-xs text-tertiary">Still checking the data…</p>
        ) : null}
      </div>
    </section>
  );
}
