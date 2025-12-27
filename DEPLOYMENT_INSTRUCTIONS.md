# Deployment Instructions for Physical AI & Humanoid Robotics Course

## Files to Deploy

The built site is located in the `docs/build/` directory. You need to deploy all files from this directory to GitHub Pages.

## Manual Deployment Steps

1. **Build the site** (already done):
   ```bash
   cd docs
   npm run build
   ```

2. **Deploy to GitHub Pages manually**:
   - Go to your GitHub repository: https://github.com/physical-ai-course/book
   - Navigate to Settings → Pages
   - Under "Source", select "Deploy from a branch"
   - Choose the `gh-pages` branch and `/` (root) folder
   - Click "Save"

3. **Create the gh-pages branch with built files**:
   ```bash
   # Navigate to the build directory
   cd docs/build

   # Create a new branch for GitHub Pages
   git init
   git remote add origin https://github.com/physical-ai-course/book.git
   git checkout -b gh-pages

   # Add all built files
   git add .
   git commit -m "Deploy site to GitHub Pages"

   # Push to the gh-pages branch
   git push -f origin gh-pages
   ```

## Alternative: Using GitHub Actions (Recommended)

Create `.github/workflows/deploy.yml` in your main repository with this content:

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: 18
          cache: npm
          cache-dependency-path: docs/package-lock.json
      - name: Install dependencies
        run: npm install
        working-directory: docs
      - name: Build website
        run: npm run build
        working-directory: docs
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs/build
          publish_branch: gh-pages
```

## Site Location

Once deployed, your site will be available at:
https://physical-ai-course.github.io/book/

## Important Notes

- The site contains comprehensive documentation for the Physical AI & Humanoid Robotics course
- Includes 6 main modules covering ROS2, Gazebo, NVIDIA Isaac, and conversational robotics
- Features weekly breakdowns, hardware requirements, and assessment guidelines
- All content is properly structured and navigable