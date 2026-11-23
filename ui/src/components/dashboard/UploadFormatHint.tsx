// The one line under the dropzone that says what can be uploaded, and the
// formats the backend knows but defers, each with its reason, so a user who
// reaches for one learns why before the server answers 415.
import { describeAccepted, type UploadFormats } from "@/lib/upload-formats";

export function UploadFormatHint({
  formats,
  limitsText,
}: {
  formats: UploadFormats;
  limitsText: string;
}) {
  return (
    <>
      <p className="mt-1 text-xs text-tertiary" data-testid="upload-accepted">
        {describeAccepted(formats)} ({limitsText})
      </p>
      {formats.deferred.length > 0 && (
        <p className="mt-1 text-xs text-quaternary" data-testid="upload-deferred">
          Not yet:{" "}
          {formats.deferred.map((entry, index) => (
            <span key={entry.format}>
              {index > 0 && ", "}
              <abbr
                title={entry.reason}
                className="cursor-help no-underline"
                data-format={entry.format}
              >
                {entry.format}
              </abbr>
            </span>
          ))}
        </p>
      )}
    </>
  );
}
