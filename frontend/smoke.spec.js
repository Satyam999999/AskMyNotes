import { test, expect } from '@playwright/test';
import path from 'path';

test.describe('Frontend Smoke Test', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:5175/');
  });

  test('Upload source', async ({ page }) => {
    await page.click('text=Upload source');
    // The collection ID input is the first input, wait for it
    await page.fill('text=Collection ID >> xpath=..//input', 'lecture-2-notes');
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(path.resolve('../Lecture 2.pdf'));
    await page.click('button:has-text("Upload and index")');
    await expect(page.locator('.output-card h3')).toContainText('Upload complete', { timeout: 30000 });
  });

  test('Ask your notes', async ({ page }) => {
    await page.click('text=Ask your notes');
    await page.fill('text=Collection ID >> xpath=..//input', 'lecture-2-notes');
    await page.fill('text=Question >> xpath=..//textarea', 'what is attention mechanism?');
    await page.click('button:has-text("Ask notebook")');
    await expect(page.locator('.output-card h3')).toContainText('Answer', { timeout: 30000 });
    await expect(page.locator('.output-card p.answer-copy')).toBeVisible();
  });

  test('Revision sheet', async ({ page }) => {
    await page.click('text=Revision sheet');
    await page.fill('text=Collection ID >> xpath=..//input', 'lecture-2-notes');
    await page.fill('text=Topic >> xpath=..//input', 'attention mechanism');
    await page.click('button:has-text("Generate revision sheet")');
    await expect(page.locator('.output-card h3')).toContainText('Revision sheet', { timeout: 30000 });
  });

  test('Auto quiz', async ({ page }) => {
    await page.click('text=Auto quiz');
    await page.fill('text=Collection ID >> xpath=..//input', 'lecture-2-notes');
    await page.fill('text=Topic >> xpath=..//input', 'attention mechanism');
    await page.click('button:has-text("Generate quiz")');
    await expect(page.locator('.output-card h3')).toContainText('Quiz', { timeout: 30000 });
    await expect(page.locator('.quiz-stack')).toBeVisible();
  });

  test('Explain simply', async ({ page }) => {
    await page.click('text=Explain simply');
    await page.fill('text=Collection ID >> xpath=..//input', 'lecture-2-notes');
    await page.fill('text=Concept >> xpath=..//input', 'attention mechanism');
    await page.click('button:has-text("Explain simply")');
    await expect(page.locator('.output-card h3')).toContainText('Simple explanation', { timeout: 30000 });
  });

  test('Audio notes', async ({ page }) => {
    await page.click('text=Audio notes');
    await page.fill('text=Collection ID >> xpath=..//input', 'lecture-2-notes');
    await page.fill('text=Topic >> xpath=..//input', 'attention mechanism');
    await page.click('button:has-text("Create audio notes")');
    await expect(page.locator('.output-card h3')).toContainText('Audio generated', { timeout: 60000 });
    await expect(page.locator('audio.native-audio')).toBeVisible();
  });

  test('Night before exam', async ({ page }) => {
    await page.click('text=Night before exam');
    await page.fill('text=Collection ID >> xpath=..//input', 'lecture-2-notes');
    await page.fill('text=Subject >> xpath=..//input', 'transformer architectures');
    await page.click('button:has-text("Generate cheat sheet")');
    await expect(page.locator('.output-card h3')).toContainText('Night-before sheet', { timeout: 30000 });
  });

  test('Flashcards', async ({ page }) => {
    await page.click('text=Flashcards');
    await page.fill('text=Collection ID >> xpath=..//input', 'lecture-2-notes');
    await page.fill('text=Topic >> xpath=..//input', 'attention mechanism');
    await page.click('button:has-text("Generate flashcards")');
    await expect(page.locator('.output-card h3')).toContainText('Flashcards', { timeout: 30000 });
    await expect(page.locator('.flashcards-wrap')).toBeVisible();
  });

  test('Smart highlights', async ({ page }) => {
    await page.click('text=Smart highlights');
    await page.fill('text=Collection ID >> xpath=..//input', 'lecture-2-notes');
    await page.fill('text=Topic >> xpath=..//input', 'attention mechanism');
    await page.click('button:has-text("Score highlights")');
    await expect(page.locator('.output-card h3')).toContainText('Highlights', { timeout: 30000 });
    await expect(page.locator('.highlights-list')).toBeVisible();
  });
});
