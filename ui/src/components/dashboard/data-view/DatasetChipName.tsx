import { readableDatasetName } from "@/lib/upload-review";

/** Label only: dataset IDs and selection remain owned by the analysis route. */
export function DatasetChipName({ name, sheetName, sourceWorkbook }: {
  name: string;
  sheetName?: string | null;
  sourceWorkbook?: string | null;
}) {
  const label = sourceWorkbook && sheetName
    ? sheetName
    : readableDatasetName(name.split("/").filter(Boolean).pop() ?? name);
  return <span className="truncate" title={name}>{label}</span>;
}