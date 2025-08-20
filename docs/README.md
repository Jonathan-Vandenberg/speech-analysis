# 📚 Speech Analysis API Documentation

This directory contains the interactive API documentation hosted on GitHub Pages.

## 🌐 Live Documentation

Visit the live documentation at: **https://your-username.github.io/audio-analysis**

## 📁 Files Overview

```
docs/
├── index.html          # Main Swagger UI documentation page
├── openapi.json        # OpenAPI 3.0 specification (auto-generated)
├── api-stats.json      # API statistics (auto-generated)
├── _config.yml         # GitHub Pages configuration
└── README.md           # This file
```

## 🔄 Automatic Updates

The documentation is automatically updated when:

1. **Code changes**: Any changes to the `main/` directory trigger a rebuild
2. **Documentation changes**: Updates to files in `docs/` trigger a rebuild
3. **Manual trigger**: You can manually trigger the workflow from GitHub Actions

### Workflow Process

1. 🔄 **Checkout code** from the repository
2. 🐍 **Setup Python** environment
3. 📦 **Install dependencies** from `requirements.txt`
4. 📄 **Generate OpenAPI spec** using `generate_openapi.py`
5. 🎯 **Validate** the generated specification
6. 📊 **Generate API statistics** for metadata
7. 🚀 **Deploy to GitHub Pages** (on main branch)
8. 💬 **Comment on PRs** with preview information

## 🛠️ Local Development

To test the documentation locally:

### Option 1: Simple HTTP Server

```bash
cd docs
python -m http.server 8080
```

Then open: http://localhost:8080

### Option 2: Jekyll (GitHub Pages simulation)

```bash
# Install Jekyll
gem install jekyll bundler

# Create Gemfile if not exists
echo "source 'https://rubygems.org'" > Gemfile
echo "gem 'github-pages', group: :jekyll_plugins" >> Gemfile

# Install dependencies
bundle install

# Serve locally
bundle exec jekyll serve

# Open http://localhost:4000/audio-analysis
```

### Option 3: Docker

```bash
# From the repository root
docker run --rm -v "$PWD/docs:/srv/jekyll" -p 4000:4000 jekyll/jekyll:latest jekyll serve --watch --incremental
```

## 🎨 Customization

### Styling

The documentation uses custom CSS in `index.html` for:
- 🎨 **Modern gradient header**
- 📱 **Responsive design**
- 🃏 **Information cards**
- 💻 **Code highlighting**
- 🎯 **Custom Swagger UI theme**

### Content

To update the documentation content:

1. **API Description**: Edit the `description` in `main/app.py`
2. **Quick Start**: Modify the quick start section in `index.html`
3. **Examples**: Update examples in the OpenAPI spec generation script
4. **Contact Info**: Update contact details in `main/app.py`

## 📊 API Statistics

The `api-stats.json` file contains:

```json
{
  "generated_at": "2024-01-15T10:30:00Z",
  "total_endpoints": 7,
  "version": "1.0.0",
  "title": "Speech Analysis API",
  "endpoints": [
    "/healthz",
    "/api/admin/keys",
    "/analyze/pronunciation",
    "..."
  ]
}
```

## 🔗 Related Links

- **🏠 Main Repository**: https://github.com/your-username/audio-analysis
- **🎯 Live API**: https://api.example.com
- **🔧 Admin Interface**: https://admin.example.com
- **📖 FastAPI Docs**: https://api.example.com/docs
- **📝 ReDoc**: https://api.example.com/redoc

## 🚀 Deployment Checklist

Before deploying to production:

- [ ] Update contact information in `_config.yml`
- [ ] Replace example URLs with actual URLs
- [ ] Set up custom domain (optional)
- [ ] Configure Google Analytics (optional)
- [ ] Test all documentation links
- [ ] Verify API examples work with live endpoints

## 🎯 Features

### Interactive Documentation
- ✅ **Try It Out**: Test API endpoints directly from the documentation
- ✅ **Authentication**: Built-in API key authentication testing
- ✅ **Response Examples**: Real response examples with detailed schemas
- ✅ **Error Codes**: Comprehensive error code documentation

### Developer Experience
- ✅ **Search**: Full-text search across all endpoints
- ✅ **Deep Linking**: Direct links to specific endpoints
- ✅ **Mobile Friendly**: Responsive design for all devices
- ✅ **Dark Mode**: Automatic dark mode support

### Automation
- ✅ **Auto-Generated**: Always in sync with the latest code
- ✅ **Validation**: Automatic validation of OpenAPI specs
- ✅ **Statistics**: Track API growth and changes over time
- ✅ **PR Previews**: Automatic documentation previews on pull requests

## 🆘 Troubleshooting

### Documentation Not Updating

1. Check the GitHub Actions workflow status
2. Verify the `generate_openapi.py` script runs without errors
3. Ensure GitHub Pages is enabled in repository settings
4. Check for any Jekyll build errors

### Local Development Issues

1. Ensure Python dependencies are installed: `pip install -r requirements.txt`
2. Check that the OpenAPI spec is valid JSON
3. Verify HTTP server is running on the correct port
4. Clear browser cache if changes aren't visible

### GitHub Pages Not Working

1. Go to repository **Settings** → **Pages**
2. Set source to "GitHub Actions" or "Deploy from a branch"
3. Select the `gh-pages` branch if using branch deployment
4. Wait a few minutes for deployment to complete

## 🤝 Contributing

To contribute to the documentation:

1. Fork the repository
2. Make your changes to the documentation files
3. Test locally using one of the methods above
4. Submit a pull request
5. The documentation will be automatically previewed in the PR

---

*This documentation is automatically generated and deployed using GitHub Actions and GitHub Pages.*
