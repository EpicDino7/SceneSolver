# Environment Variables Setup

This Flask backend uses environment variables to securely manage configuration settings, particularly sensitive information like database credentials.

## Quick Setup

1. **Copy the example environment file:**

   ```bash
   cp .env.example .env
   ```

2. **Edit the `.env` file with your actual values:**

   ```bash
   # MongoDB Configuration
   MONGODB_URL=mongodb+srv://your_username:your_password@your_cluster.mongodb.net/your_database?retryWrites=true&w=majority&appName=YourAppName
   MONGODB_DB_NAME=crime_db
   MONGODB_COLLECTION_NAME=predictions

   # Flask Configuration
   FLASK_ENV=development
   UPLOAD_FOLDER=uploads
   FRAMES_FOLDER=frames
   ```

3. **Install the required dependency:**
   ```bash
   pip install python-dotenv
   ```
   (Already included in `requirements.txt`)

## Environment Variables

### Required Variables

- **`MONGODB_URL`**: Your MongoDB connection string
  - **IMPORTANT**: This contains sensitive credentials
  - Format: `mongodb+srv://username:password@cluster.mongodb.net/database?options`

### Optional Variables (with defaults)

- **`MONGODB_DB_NAME`**: Database name (default: `crime_db`)
- **`MONGODB_COLLECTION_NAME`**: Collection name (default: `predictions`)
- **`UPLOAD_FOLDER`**: Directory for uploaded files (default: `uploads`)
- **`FRAMES_FOLDER`**: Directory for extracted video frames (default: `frames`)
- **`FLASK_ENV`**: Flask environment mode (default: `development`)

## Security Notes

🔒 **IMPORTANT**:

- The `.env` file is already included in `.gitignore` and will NOT be committed to Git
- Never commit your `.env` file to version control
- Keep your MongoDB credentials secure
- Use different `.env` files for development, testing, and production

## Usage

The application will automatically load environment variables from the `.env` file when it starts. If the `.env` file is missing or `MONGODB_URL` is not set, the application will show an error message.

## Troubleshooting

If you see "MONGODB_URL environment variable not set":

1. Make sure you have a `.env` file in the `backendflask/` directory
2. Check that `MONGODB_URL` is properly set in your `.env` file
3. Ensure there are no extra spaces around the `=` sign
4. Restart the Flask application after making changes
