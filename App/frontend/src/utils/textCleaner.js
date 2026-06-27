export const cleanWikiText = (text) => {
  if (!text) return "";
  let cleaned = text;
  
  // Remove JSON blocks related to images
  cleaned = cleaned.replace(/\{[^}]*smallUrl[^}]*\}/gi, "");
  
  // Remove stray HTML tags
  cleaned = cleaned.replace(/<[^>]*>/gi, "");
  
  // Remove specific WikiHow/WikiLingua scrape artifacts
  cleaned = cleaned.replace(/mw-parser-output/gi, "");
  cleaned = cleaned.replace(/\.big\b/gi, "");
  cleaned = cleaned.replace(/\.small\b/gi, "");
  cleaned = cleaned.replace(/\.ensing\b/gi, "");
  
  // Remove hallucinated JSON punctuation like \": or ":
  cleaned = cleaned.replace(/\\?"\s*:/g, "");
  
  // Fix punctuation spacing (e.g., "word ." -> "word.")
  cleaned = cleaned.replace(/\s+\./g, ".");
  cleaned = cleaned.replace(/\s+,/g, ",");
  
  // Normalize multiple spaces into a single space
  cleaned = cleaned.replace(/\s+/g, " ").trim();
  
  return cleaned;
};
