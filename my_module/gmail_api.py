import os
import pickle
import base64
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# If modifying these SCOPES, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', 'https://www.googleapis.com/auth/spreadsheets']

# The ID and range of a sample spreadsheet.
SAMPLE_SPREADSHEET_ID = '11A9DW6er6wVfxYYNLy1tA_wzgOE-n85uOf4DiaGrd0k'
SAMPLE_RANGE_NAME = 'Hoja 1!A:D'

def main():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=8000)
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)

    # Here, create both services (Gmail and Sheets)
    gmail_service = build('gmail', 'v1', credentials=creds)
    sheets_service = build('sheets', 'v4', credentials=creds)

    def get_email_data(service):
        results = service.users().messages().list(userId='me', labelIds=['INBOX'], q="is:unread").execute()
        messages = results.get('messages', [])

        # Get the first message
        message = messages[0]
        msg = service.users().messages().get(userId='me', id=message['id']).execute()

        # Get the email data payload
        payload = msg['payload']

        parts = payload.get('parts')

        data_text = None
        data_html = None

        for part in parts:
            mimeType = part.get('mimeType')
            body = part.get('body')
            data = body.get('data')

            if part.get('parts'):
                # If this part contains further parts, recursively extract the text/plain part
                for subpart in part.get('parts'):
                    if subpart.get('mimeType') == 'text/plain':
                        data_text = subpart.get('body').get('data')
                    elif subpart.get('mimeType') == 'text/html':
                        data_html = subpart.get('body').get('data')
            if mimeType == "text/plain":
                data_text = data
            if mimeType == "text/html":
                data_html = data

            if data_text:
                body_text = base64.urlsafe_b64decode(data_text).decode()
            elif data_html:
                body_text = base64.urlsafe_b64decode(data_html).decode()

        name = None
        email = None
        subject = None

        for header in payload.get('headers'):
            header_name = header.get('name')
            value = header.get('value')
            if header_name.lower() == 'from':
                name, email = value.split(' <')
                email = email.replace('>', '')  # remove the trailing '>'
            elif header_name.lower() == 'subject':
                subject = value

        return {'name': name, 'email': email, 'subject': subject, 'body': body_text}

    # Fetch the email data from the Gmail API
    email_data = get_email_data(gmail_service)  # Here you pass the service as an argument to the function

    # Use the email data to construct the values array
    values = [[
        email_data['name'], 
        email_data['email'], 
        email_data['subject'], 
        email_data['body'], 
    ]]


    # Call the Sheets API to append the values to the spreadsheet
    body = {'values': values}
    print("About to append data to sheet...")
    # result = sheets_service.spreadsheets().values().get(
    #     spreadsheetId=SAMPLE_SPREADSHEET_ID,
    #     range=SAMPLE_RANGE_NAME
    # ).execute()

    # print(result)



    result = sheets_service.spreadsheets().values().append(
        spreadsheetId=SAMPLE_SPREADSHEET_ID,
        range=SAMPLE_RANGE_NAME,
        valueInputOption='RAW',
        body=body
    ).execute()
    print("Data append operation comleted.")    
    
    spreadsheet = sheets_service.spreadsheets().get(spreadsheetId=SAMPLE_SPREADSHEET_ID).execute()
    sheets = spreadsheet.get('sheets', '')
    titles = [sheet.get('properties', {}).get('title') for sheet in sheets]
    print('Sheet Titles:', titles)

    print(f'{result.get("updates").get("updatedCells")} cells appended.')

if __name__ == '__main__':
    main()
