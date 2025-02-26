using System;
using System.IO;
using System.Text.RegularExpressions;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Collections.Generic;
using HtmlAgilityPack;
class Program
{
    static void Main(string[] args)
    {
        string FilePath = "data.json";
        string OutputPath = "cleaned_json.json";

        string JsonContent = File.ReadAllText(FilePath);
        List<PageData> data = JsonSerializer.Deserialize<List<PageData>>(JsonContent);

        for (int i = 0; i < data.Count; i++)
        {
            data[i].Content = CleanText(data[i].Content);
            Console.WriteLine($"{i}. Cleared Content");
        }

        string cleanedJson = JsonSerializer.Serialize(data, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(OutputPath, cleanedJson);

        Console.WriteLine($"Cleaned data saved to {OutputPath}");

    }

    static string CleanText(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return text;
        }

        text = Regex.Replace(text, @"[^\u0000-\u007F]+", string.Empty);
        text = Regex.Replace(text, @"\s+", " ").Trim();
        string[] UiElements = { "Table of contents", "Read in English", "Add to Collections","Edit", "Share via", "Expand table", "Print", "Save", "Feedback" };

        foreach (var element in UiElements)
        {
            text = Regex.Replace(text, element, "", RegexOptions.IgnoreCase);
        }

        text = Regex.Replace(text, @"\{""@context"".*?\}\]", "", RegexOptions.Singleline);
        text = RemoveHTML(RemoveHTML(text));

        return text;

    }

    static string RemoveHTML(string input)
    {
        HtmlDocument doc = new HtmlDocument();
        doc.LoadHtml(input);
        return doc.DocumentNode.InnerText;
    }
}

class PageData
{
    [JsonPropertyName("url")]
    public string Url { get; set; }

    [JsonPropertyName("title")]
    public string Title { get; set; }

    [JsonPropertyName("content")]
    public string Content { get; set; }
}