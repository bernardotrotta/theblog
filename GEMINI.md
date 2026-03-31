# GEMINI.md - Project Context

## Project Overview
This is a Hugo-based blog project using the `hello-friend-ng` theme. It appears to have been migrated from a static site structure to Hugo, with a focus on simplicity and modern design. The project is configured for deployment to GitHub Pages using GitHub Actions.

- **Main Technologies:** [Hugo](https://gohugo.io/) (Extended version), [Dart Sass](https://sass-lang.com/dart-sass), [Go](https://go.dev/), [Node.js](https://nodejs.org/).
- **Theme:** `hello-friend-ng` (managed as a Git submodule).
- **Hosting:** GitHub Pages.
- **Architecture:** Static site generated from Markdown content in `content/` (currently empty or untracked) and custom layouts in `layouts/`.

## Building and Running
The following commands are typically used for local development and production builds:

- **Local Development:** 
  - `hugo server -D` to run the development server with draft posts enabled.
  - `hugo server` to run the development server normally.
- **Production Build:**
  - `hugo --gc --minify` to build a minified, production-ready site in the `public/` directory.
- **CI/CD:** 
  - Managed by `.github/workflows/hugo.yaml`, which automates the build and deployment to GitHub Pages upon pushes to the `main` branch.

## Directory Structure
- `archetypes/`: Contains templates for new content files (e.g., `default.md`).
- `assets/`: Custom assets for the site (e.g., Sass/SCSS, JS).
- `content/`: Where the site's Markdown content (posts, pages) is stored.
- `data/`: Custom data files used by Hugo for site generation.
- `i18n/`: Internationalization files for multi-language support.
- `layouts/`: Custom HTML layouts and overrides for the theme.
- `public/`: The directory where the built static site is generated.
- `resources/`: Cached and generated assets (e.g., processed images, compiled SCSS).
- `static/`: Static files that are copied as-is to the root of the site (e.g., images like `mia-foto.jpg`).
- `themes/`: Contains the `hello-friend-ng` theme submodule.

## Development Conventions
- **Content Creation:** Use `hugo new posts/my-post.md` to create new blog posts using the default archetype.
- **Theme Customization:** Prefer overriding theme files by placing them in the root `layouts/` or `assets/` directories instead of modifying the theme submodule directly.
- **Deployment:** Automatically triggered by pushes to the `main` branch. Ensure submodules are initialized and updated before building locally.
