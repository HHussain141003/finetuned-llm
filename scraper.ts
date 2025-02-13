import { chromium } from "playwright";
import * as fs from "fs-extra";

const BASE_URL = "https://learn.microsoft.com/en-us/docs/";
const jsonFile = "data.json";
const MAX_WORKERS = 10;
const queue: string[] = [BASE_URL];
const visitedUrls = new Set<string>();

if (!fs.existsSync(jsonFile)) {
    fs.writeFileSync(jsonFile, "[]", "utf8");
} else {
    const existingData = JSON.parse(fs.readFileSync(jsonFile, "utf8"));
    existingData.forEach((entry: { url: string }) => {
        visitedUrls.add(entry.url);
    });
}

async function scrapePage(url: string, browser: any) {
    if (visitedUrls.has(url)) return;
    visitedUrls.add(url);
    console.log(`Scraping: ${url}`);

    const page = await browser.newPage();
    try {
        await page.goto(url, { waitUntil: "domcontentloaded", timeout: 60000 });
        await page.waitForSelector('main[role="main"]', { timeout: 60000 });

        const title = await page.title();
        const content = await page.textContent('main[role="main"]') || "";

        if (content) {
            const existingData = JSON.parse(fs.readFileSync(jsonFile, "utf8"));
            existingData.push({ url, title, content });
            fs.writeFileSync(jsonFile, JSON.stringify(existingData, null, 2));
        }

        const links = await page.$$eval("a", (anchors) =>
            anchors.map((a) => a.href).filter((href) =>
                href.includes("learn.microsoft.com/en-us/")
            )
        );

        for (const link of links) {
            if (!visitedUrls.has(link) && !queue.includes(link)) {
                queue.push(link);
            }
        }
    } catch (error) {
        console.error(`Error scraping ${url}:`, error);
    } finally {
        await page.close();
    }
}

async function worker(browser: any) {
    while (queue.length > 0) {
        const url = queue.shift();
        if (url) {
            try {
                await scrapePage(url, browser);
            } catch (error) {
                console.error(`Worker error on ${url}:`, error);
            }
        }
    }
}

(async () => {
    const browser = await chromium.launch({ headless: true });

    const workers = Array.from({ length: MAX_WORKERS }, () => worker(browser));

    await Promise.all(workers);

    await browser.close();
})();