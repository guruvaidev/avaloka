import { createServerFn } from "@tanstack/react-start";

type SalesInquiry = {
  name: string;
  companyName: string;
  email: string;
  phone: string;
  message?: string;
};

function escapeHtml(value: string) {
  return value.replace(/[&<>"']/g, (char) => {
    const entities: Record<string, string> = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    };
    return entities[char];
  });
}

export const submitSalesInquiry = createServerFn({ method: "POST" })
  .inputValidator((input: SalesInquiry) => {
    const name = input.name.trim();
    const companyName = input.companyName.trim();
    const email = input.email.trim().toLowerCase();
    const phone = input.phone.trim();
    const message = (input.message ?? "").trim();

if (message.length > 2000) {
  throw new Error("Message must be 2000 characters or fewer.");
}

    if (!name || !companyName || !email || !phone) {
      throw new Error("Please complete all fields.");
    }

    if (name.length > 100 || companyName.length > 150 ||
        email.length > 254 || phone.length > 30) {
      throw new Error("One or more fields are too long.");
    }

    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      throw new Error("Please enter a valid email address.");
    }

    return { name, companyName, email, phone, message  };
  })
  .handler(async ({ data }) => {
    const supabaseUrl = process.env["SUPABASE_INCLUSTER_URL"];
    const token =
      process.env["INTERNAL_MAIL_TOKEN"] ||
      process.env["PRIMARY_SUPABASE_SERVICE_ROLE_KEY"];

    
    const recipient =process.env["EMAIL_FROM"];


    if (!supabaseUrl || !token || !recipient) {
      throw new Error("Sales email service is not configured.");
    }

   
const LOGO = "https://storage.googleapis.com/avaloka-public-assets/Logo.svg";

const html = `
<!doctype html>
<html>
<body style="margin:0;padding:0;background:#ffffff">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#ffffff">
  <tr>
    <td align="center" style="padding:24px 16px">
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
        style="max-width:560px;font-family:Arial,sans-serif;color:#374151">

        <!-- Header Logo -->
        <tr>
          <td>
            <table role="presentation" cellpadding="0" cellspacing="0">
              <tr>
                <td style="vertical-align:middle">
                  <img src="${LOGO}" width="28" height="28" alt="Avaloka AI"
                    style="display:block;border:0;outline:none;text-decoration:none" />
                </td>
                <td style="vertical-align:middle;padding-left:10px;font-size:18px;font-weight:700;color:#111">
                  Avaloka AI
                </td>
              </tr>
            </table>
          </td>
        </tr>

        <!-- Title -->
        <tr>
          <td style="padding-top:24px;font-size:22px;font-weight:700;color:#111">
            New Avaloka Enterprise Sales Inquiry
          </td>
        </tr>

        <!-- Introduction -->
        <tr>
          <td style="padding-top:20px;font-size:15px;line-height:24px">
            Hi,<br/><br/>
            Someone has submitted an enterprise sales inquiry through the Avaloka website.
            Here are the details:
          </td>
        </tr>

        <!-- Inquiry Details -->
        <tr>
          <td style="padding-top:16px">
            <table role="presentation" cellpadding="0" cellspacing="0" width="100%">
              <tr>
                <td style="padding:6px 0;color:#6b7280;font-size:14px;width:120px;vertical-align:top">
                  Name
                </td>
                <td style="padding:6px 0;color:#111;font-size:14px">
                  ${escapeHtml(data.name)}
                </td>
              </tr>

              <tr>
                <td style="padding:6px 0;color:#6b7280;font-size:14px;width:120px;vertical-align:top">
                  Company
                </td>
                <td style="padding:6px 0;color:#111;font-size:14px">
                  ${escapeHtml(data.companyName)}
                </td>
              </tr>

              <tr>
                <td style="padding:6px 0;color:#6b7280;font-size:14px;width:120px;vertical-align:top">
                  Email
                </td>
                <td style="padding:6px 0;color:#111;font-size:14px">
                  <a href="mailto:${escapeHtml(data.email)}"
                    style="color:#1565ef;text-decoration:none">
                    ${escapeHtml(data.email)}
                  </a>
                </td>
              </tr>

              <tr>
                <td style="padding:6px 0;color:#6b7280;font-size:14px;width:120px;vertical-align:top">
                  Phone
                </td>
                <td style="padding:6px 0;color:#111;font-size:14px">
                  <a href="tel:${escapeHtml(data.phone.replace(/[^\\d+]/g, ""))}"
                    style="color:#111;text-decoration:none">
                    ${escapeHtml(data.phone)}
                  </a>
                </td>
              </tr>
              <tr>
                <td style="padding:6px 0;color:#6b7280;font-size:14px;width:120px;vertical-align:top">
                  Message
                </td>
                <td style="padding:6px 0;color:#111;font-size:14px;line-height:22px;white-space:pre-wrap;overflow-wrap:anywhere">
                  ${escapeHtml(data.message || "No message provided")}
                </td>
              </tr>
            </table>
          </td>
        </tr>

        <!-- Footer -->
        <tr>
          <td style="padding-top:32px;font-size:12px;color:#6b7280">
            This email was sent from the enterprise sales inquiry form at
            <a href="https://dev.avaloka.ai/contact-us" style="color:#1565ef">
              avaloka.ai
            </a>.
          </td>
        </tr>

        <tr>
          <td style="padding-top:12px;font-size:12px;line-height:20px;color:#6b7280">
            &copy; ${new Date().getFullYear()} Avaloka AI,
            18273 Sun Maiden Ct, San Diego, CA 92127, USA
          </td>
        </tr>

        <!-- Bottom Logo -->
        <tr>
          <td style="padding-top:24px">
            <table role="presentation" cellpadding="0" cellspacing="0">
              <tr>
                <td style="vertical-align:middle">
                  <img src="${LOGO}" width="20" height="20" alt="Avaloka AI"
                    style="display:block;border:0;outline:none;text-decoration:none" />
                </td>
                <td style="vertical-align:middle;padding-left:8px;font-size:15px;font-weight:700;color:#111">
                  Avaloka AI
                </td>
              </tr>
            </table>
          </td>
        </tr>

      </table>
    </td>
  </tr>
</table>
</body>
</html>
`;


    const response = await fetch(
      `${supabaseUrl.replace(/\/+$/, "")}/functions/v1/send-smtp-email`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          to: recipient,
          subject: `Enterprise sales inquiry — ${data.companyName}`,
          html,
          text: [
            "New Avaloka Enterprise Sales Inquiry",
            `Name: ${data.name}`,
            `Company: ${data.companyName}`,
            `Email: ${data.email}`,
            `Phone: ${data.phone}`,
          ].join("\n"),
        }),
      },
    );

    if (!response.ok) {
      console.error(
        "[submitSalesInquiry] SMTP service failed:",
        response.status,
        await response.text(),
      );
      throw new Error("Could not send your inquiry. Please try again.");
    }

    return { ok: true };
  });