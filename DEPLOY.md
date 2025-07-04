# 🚀 Deploying BoardRAG to Render.com

## Quick Deployment Steps

### Option 1: Using render.yaml (Recommended)

1. **Fork/Clone this repository** to your GitHub account

2. **Sign up for Render.com** at https://render.com

3. **Connect your GitHub repository**:
   - Click "New +" → "Web Service"
   - Connect your GitHub account
   - Select this repository

4. **Render will automatically detect the `render.yaml` file**
   - Review the configuration
   - Click "Create Web Service"

5. **Set Environment Variables** in Render Dashboard:
   ```
   OPENAI_API_KEY=sk-your-actual-openai-api-key
   LLM_PROVIDER=openai
   CHROMA_PATH=./chroma
   DATA_PATH=./data
   ENABLE_CHROMA_TELEMETRY=False
   ```

6. **Deploy** - Render will automatically build and deploy your app!

### Option 2: Manual Configuration

If you prefer manual setup:

1. **Create New Web Service** on Render
2. **Build Command**: `pip install -r requirements.txt`
3. **Start Command**: `python app.py`
4. **Environment**: `Python 3`
5. **Plan**: Free (or Starter for better performance)

## Required Environment Variables

Set these in your Render service dashboard:

| Variable | Value | Required |
|----------|-------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | ✅ Yes |
| `LLM_PROVIDER` | `openai` | ✅ Yes |
| `CHROMA_PATH` | `./chroma` | ✅ Yes |
| `DATA_PATH` | `./data` | ✅ Yes |
| `ENABLE_CHROMA_TELEMETRY` | `False` | ✅ Yes |
| `ANTHROPIC_API_KEY` | Your Claude API key | ❌ Optional |

## Getting Your OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Copy the key and add it to Render environment variables

## After Deployment

1. **Upload PDFs**: Use the "📤 Add New Game" section to upload board game rulebooks
2. **Query Games**: Select a game and start asking questions!
3. **Dark Mode**: Toggle with the 🌙 button

## Troubleshooting

- **Build fails**: Check that all dependencies in `requirements.txt` are correct
- **App won't start**: Verify all required environment variables are set
- **No games available**: Upload some PDF rulebooks first
- **API errors**: Check your OpenAI API key is valid and has credits

## Free Tier Limitations

Render's free tier includes:
- 750 hours/month runtime
- 1GB persistent disk
- Sleeps after 15 minutes of inactivity
- Slower startup times

For production use, consider upgrading to a paid plan.

## Support

If you encounter issues:
1. Check Render service logs
2. Verify environment variables
3. Ensure your OpenAI API key has sufficient credits 