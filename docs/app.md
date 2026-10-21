---
title: Web App
author: Tom Stanton
icon: lucide/app-window
hide:
  - navigation
  - toc
---

<style>
  .md-main__inner {
    margin-top: 0.5rem;
    max-width: 100% !important;
  }
  .md-content__inner {
    margin: 0 1rem !important;
    padding-top: 0 !important;
    max-width: 100% !important;
  }
  .md-content__inner > h1 {
    display: none;
  }
  .seroepi-app-toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
    padding: 0.4rem 0.75rem;
    border-radius: 0.4rem;
    background: var(--md-code-bg-color);
    font-size: 0.85rem;
  }
  .seroepi-app-links {
    display: flex;
    gap: 0.75rem;
    align-items: center;
  }
  .seroepi-app-frame {
    width: 100%;
    height: calc(100vh - 9.5rem);
    min-height: 760px;
    border: 1px solid var(--md-default-fg-color--lightest);
    border-radius: 0.5rem;
    background: var(--md-default-bg-color);
  }
</style>

<div class="seroepi-app-toolbar">
  <span>
    <strong>SeroEpi Interactive Dashboard</strong> — Powered by PyShiny &amp; JAX/NumPyro on Posit Connect Cloud
  </span>
  <div class="seroepi-app-links">
    <a href="https://seroverse-seroepi.share.connect.posit.cloud" target="_blank" rel="noopener" class="md-button md-button--primary" style="padding: 0.25rem 0.75rem; font-size: 0.8rem;">
      Open in New Tab ↗
    </a>
  </div>
</div>

<iframe
  src="https://seroverse-seroepi.share.connect.posit.cloud"
  class="seroepi-app-frame"
  title="SeroEpi Interactive Web Application"
  allow="clipboard-write; fullscreen"
  loading="lazy"
></iframe>
