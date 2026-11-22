import datetime
import os

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain_core.tools import tool

# Permission to read AND create events
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
TIMEZONE = "Asia/Colombo"


def get_calendar_service():
    """Logs in to Google (first time opens a browser) and returns the Calendar API."""
    creds = None
    if os.path.exists("token.json"):                     # saved login from last time
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())                    # renew an expired login
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)       # opens browser to sign in
        with open("token.json", "w") as f:
            f.write(creds.to_json())                    # save login for next time

    return build("calendar", "v3", credentials=creds)


@tool
def create_event(title: str, date: str, start_time: str, duration_minutes: int = 60) -> str:
    """Creates a new event in the user's Google Calendar.
    date must be YYYY-MM-DD, start_time must be 24-hour HH:MM.
    Example: title='Study group', date='2026-11-15', start_time='14:00', duration_minutes=60."""
    try:
        start = datetime.datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M")
    except ValueError:
        return "Invalid date or time. Use date YYYY-MM-DD and time HH:MM (24-hour)."
    end = start + datetime.timedelta(minutes=int(duration_minutes))

    event = {
        "summary": title,
        "start": {"dateTime": start.isoformat(), "timeZone": TIMEZONE},
        "end": {"dateTime": end.isoformat(), "timeZone": TIMEZONE},
    }
    created = get_calendar_service().events().insert(calendarId="primary", body=event).execute()
    return (f"Event '{title}' created on {date} from {start:%H:%M} to {end:%H:%M}. "
            f"Link: {created.get('htmlLink')}")


@tool
def get_today_schedule() -> str:
    """Gets all events in the user's Google Calendar for today.
    Use this when the user asks about today's schedule, meetings or plans."""
    now = datetime.datetime.now().astimezone()          # local time with +05:30
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = start_of_day + datetime.timedelta(days=1)

    result = get_calendar_service().events().list(
        calendarId="primary",
        timeMin=start_of_day.isoformat(),
        timeMax=end_of_day.isoformat(),
        singleEvents=True,
        orderBy="startTime",
    ).execute()
    events = result.get("items", [])

    if not events:
        return "You have no events today."
    lines = []
    for e in events:
        start = e["start"].get("dateTime", e["start"].get("date"))
        time_text = start[11:16] if "T" in start else "All day"
        lines.append(f"- {time_text}: {e.get('summary', '(no title)')}")
    return "Today's schedule:\n" + "\n".join(lines)


# Run this file directly once to sign in and test
if __name__ == "__main__":
    print(get_today_schedule.invoke({}))