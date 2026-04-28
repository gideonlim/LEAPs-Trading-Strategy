"""Send email with attachments via Gmail SMTP.

Used by the weekly report workflow to email the PDF performance report
and trades CSV. Requires a Gmail account with an App Password (not the
regular password — 2FA must be enabled, then generate an App Password
at https://myaccount.google.com/apppasswords).
"""
from __future__ import annotations

import logging
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

logger = logging.getLogger(__name__)

GMAIL_SMTP_HOST = "smtp.gmail.com"
GMAIL_SMTP_PORT = 587


def send_report_email(
    gmail_user: str,
    gmail_app_password: str,
    to_email: str,
    subject: str,
    body: str,
    attachments: list[Path],
) -> None:
    """Send an email with file attachments via Gmail SMTP.

    Raises on failure so the workflow step exits non-zero.
    """
    msg = MIMEMultipart()
    msg["From"] = gmail_user
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    for filepath in attachments:
        if not filepath.exists():
            logger.warning("Attachment not found, skipping: %s", filepath)
            continue

        part = MIMEBase("application", "octet-stream")
        part.set_payload(filepath.read_bytes())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename={filepath.name}",
        )
        msg.attach(part)
        logger.info("Attached: %s (%d bytes)", filepath.name, filepath.stat().st_size)

    with smtplib.SMTP(GMAIL_SMTP_HOST, GMAIL_SMTP_PORT) as server:
        server.starttls()
        server.login(gmail_user, gmail_app_password)
        server.send_message(msg)

    logger.info("Email sent to %s", to_email)
