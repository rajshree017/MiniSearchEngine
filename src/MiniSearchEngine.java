import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.io.*;
import java.net.*;
import java.util.regex.*;

// ============================================================
//  DOCUMENT CLASS
// ============================================================
class Document {
    private static AtomicInteger idCounter = new AtomicInteger(1);
    private int id;
    private String url;
    private String title;
    private String content;

    public Document(String url, String title, String content) {
        this.id      = idCounter.getAndIncrement();
        this.url     = url;
        this.title   = title;
        this.content = content;
    }

    public int getId()         { return id; }
    public String getUrl()     { return url; }
    public String getTitle()   { return title; }
    public String getContent() { return content; }

    public String[] getWords() {
        return content.toLowerCase().replaceAll("[^a-z0-9 ]", "").split("\\s+");
    }

    public void display() {
        System.out.println("  [Doc " + id + "] " + title);
        System.out.println("  URL     : " + url);
        System.out.println("  Preview : " + content.substring(0, Math.min(80, content.length())) + "...");
    }
}

// ============================================================
//  INVERTED INDEX (Custom Data Structure)
// ============================================================
class InvertedIndex {
    // word -> { docId -> frequency }
    private HashMap<String, HashMap<Integer, Integer>> index;
    private HashMap<Integer, Document> documents;

    public InvertedIndex() {
        index     = new HashMap<>();
        documents = new HashMap<>();
    }

    // Add document to index
    public synchronized void addDocument(Document doc) {
        documents.put(doc.getId(), doc);
        String[] words = doc.getWords();

        for (String word : words) {
            if (word.isEmpty()) continue;
            index.putIfAbsent(word, new HashMap<>());
            HashMap<Integer, Integer> docFreq = index.get(word);
            docFreq.put(doc.getId(), docFreq.getOrDefault(doc.getId(), 0) + 1);
        }
    }

    // Get docs containing the word
    public HashMap<Integer, Integer> getDocFrequency(String word) {
        return index.getOrDefault(word.toLowerCase(), new HashMap<>());
    }

    public Document getDocument(int id)     { return documents.get(id); }
    public int getTotalDocuments()          { return documents.size(); }
    public Set<Integer> getAllDocIds()      { return documents.keySet(); }
    public int getDocumentFrequency(String word) {
        return index.getOrDefault(word.toLowerCase(), new HashMap<>()).size();
    }
}

// ============================================================
//  TF-IDF RANKER
// ============================================================
class TFIDFRanker {
    private InvertedIndex index;

    public TFIDFRanker(InvertedIndex index) {
        this.index = index;
    }

    // TF = term frequency in doc / total words in doc
    private double computeTF(String word, Document doc) {
        String[] words = doc.getWords();
        int count = 0;
        for (String w : words) if (w.equals(word)) count++;
        return (double) count / words.length;
    }

    // IDF = log(total docs / docs containing word)
    private double computeIDF(String word) {
        int totalDocs = index.getTotalDocuments();
        int docsWithWord = index.getDocumentFrequency(word);
        if (docsWithWord == 0) return 0;
        return Math.log((double) totalDocs / docsWithWord);
    }

    // Rank documents for a query
    public List<Map.Entry<Document, Double>> rank(String query) {
        String[] queryWords = query.toLowerCase().replaceAll("[^a-z0-9 ]", "").split("\\s+");
        HashMap<Integer, Double> scores = new HashMap<>();

        for (String word : queryWords) {
            HashMap<Integer, Integer> docFreq = index.getDocFrequency(word);
            double idf = computeIDF(word);

            for (Map.Entry<Integer, Integer> entry : docFreq.entrySet()) {
                Document doc = index.getDocument(entry.getKey());
                double tf = computeTF(word, doc);
                double tfidf = tf * idf;
                scores.put(entry.getKey(), scores.getOrDefault(entry.getKey(), 0.0) + tfidf);
            }
        }

        // Sort by score descending
        List<Map.Entry<Document, Double>> ranked = new ArrayList<>();
        for (Map.Entry<Integer, Double> entry : scores.entrySet()) {
            ranked.add(new AbstractMap.SimpleEntry<>(index.getDocument(entry.getKey()), entry.getValue()));
        }
        ranked.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));
        return ranked;
    }
}

// ============================================================
//  WEB CRAWLER (Multi-threaded)
// ============================================================
class WebCrawler {
    private InvertedIndex index;
    private Set<String> visited;
    private ExecutorService threadPool;
    private static final int MAX_THREADS = 5;

    public WebCrawler(InvertedIndex index) {
        this.index      = index;
        this.visited    = Collections.synchronizedSet(new HashSet<>());
        this.threadPool = Executors.newFixedThreadPool(MAX_THREADS);
    }

    public void crawl(String url) {
        if (visited.contains(url)) return;
        visited.add(url);

        threadPool.submit(() -> {
            try {
                System.out.println("  [Crawler] Fetching: " + url);
                URL u = new URL(url);
                HttpURLConnection conn = (HttpURLConnection) u.openConnection();
                conn.setRequestMethod("GET");
                conn.setConnectTimeout(3000);
                conn.setReadTimeout(3000);
                conn.setRequestProperty("User-Agent", "MiniSearchEngine/1.0");

                BufferedReader reader = new BufferedReader(new InputStreamReader(conn.getInputStream()));
                StringBuilder html = new StringBuilder();
                String line;
                while ((line = reader.readLine()) != null) html.append(line).append(" ");
                reader.close();

                String rawHtml = html.toString();

                // Extract title
                String title = extractTag(rawHtml, "title");
                if (title.isEmpty()) title = url;

                // Strip HTML tags for content
                String content = rawHtml.replaceAll("<[^>]+>", " ").replaceAll("\\s+", " ").trim();
                if (content.length() > 500) content = content.substring(0, 500);

                Document doc = new Document(url, title, content);
                index.addDocument(doc);
                System.out.println("  [Crawler] Indexed: " + title);

            } catch (Exception e) {
                System.out.println("  [Crawler] Failed: " + url + " (" + e.getMessage() + ")");
            }
        });
    }

    private String extractTag(String html, String tag) {
        Pattern p = Pattern.compile("<" + tag + "[^>]*>(.*?)</" + tag + ">", Pattern.CASE_INSENSITIVE | Pattern.DOTALL);
        Matcher m = p.matcher(html);
        if (m.find()) return m.group(1).replaceAll("<[^>]+>", "").trim();
        return "";
    }

    public void shutdown() {
        threadPool.shutdown();
        try { threadPool.awaitTermination(10, TimeUnit.SECONDS); }
        catch (InterruptedException e) { Thread.currentThread().interrupt(); }
    }
}

// ============================================================
//  SEARCH ENGINE (Core)
// ============================================================
class SearchEngine {
    private InvertedIndex index;
    private TFIDFRanker   ranker;
    private WebCrawler    crawler;

    private static final int PAGE_SIZE = 3; // results per page

    public SearchEngine() {
        this.index   = new InvertedIndex();
        this.ranker  = new TFIDFRanker(index);
        this.crawler = new WebCrawler(index);
    }

    // Manually add document (for demo/testing)
    public void addDocument(String url, String title, String content) {
        index.addDocument(new Document(url, title, content));
    }

    // Crawl a URL
    public void crawl(String url) { crawler.crawl(url); }

    // Search with pagination
    public void search(String query, int page) {
        System.out.println("\n========== SEARCH: \"" + query + "\" (Page " + page + ") ==========");

        List<Map.Entry<Document, Double>> results = ranker.rank(query);

        if (results.isEmpty()) {
            System.out.println("No results found for: " + query);
            return;
        }

        // Pagination
        int start = (page - 1) * PAGE_SIZE;
        int end   = Math.min(start + PAGE_SIZE, results.size());
        int totalPages = (int) Math.ceil((double) results.size() / PAGE_SIZE);

        if (start >= results.size()) {
            System.out.println("No more results. Total pages: " + totalPages);
            return;
        }

        System.out.println("Found " + results.size() + " results | Page " + page + " of " + totalPages);
        System.out.println("--------------------------------------------------");

        for (int i = start; i < end; i++) {
            Map.Entry<Document, Double> entry = results.get(i);
            System.out.println("\nResult #" + (i + 1));
            entry.getKey().display();
            System.out.printf("  Score   : %.4f\n", entry.getValue());
            System.out.println("  --------------------------------------------------");
        }
    }

    public void shutdown() { crawler.shutdown(); }
    public int getTotalDocuments() { return index.getTotalDocuments(); }
}

// ============================================================
//  MAIN CLASS - Console Menu
// ============================================================
public class MiniSearchEngine {

    static Scanner      sc     = new Scanner(System.in);
    static SearchEngine engine = new SearchEngine();

    public static void main(String[] args) {
        System.out.println("\n========================================");
        System.out.println("        MINI SEARCH ENGINE");
        System.out.println("  Multi-threaded | TF-IDF | Inverted Index");
        System.out.println("========================================");

        // Load sample documents for demo
        loadSampleDocuments();

        while (true) {
            printMenu();
            System.out.print("Enter your choice: ");
            int choice;
            try { choice = Integer.parseInt(sc.nextLine().trim()); }
            catch (NumberFormatException e) { System.out.println("Invalid input!"); continue; }

            switch (choice) {
                case 1 -> searchMenu();
                case 2 -> addDocumentMenu();
                case 3 -> crawlMenu();
                case 4 -> System.out.println("\nTotal documents indexed: " + engine.getTotalDocuments());
                case 0 -> { engine.shutdown(); System.out.println("\nGoodbye!"); System.exit(0); }
                default -> System.out.println("Invalid choice!");
            }
        }
    }

    static void printMenu() {
        System.out.println("\n========== MAIN MENU ==========");
        System.out.println("  1. Search");
        System.out.println("  2. Add Document Manually");
        System.out.println("  3. Crawl a URL");
        System.out.println("  4. View Total Indexed Documents");
        System.out.println("  0. Exit");
        System.out.println("================================");
    }

    static void searchMenu() {
        System.out.print("\nEnter search query: ");
        String query = sc.nextLine().trim();
        System.out.print("Page number (default 1): ");
        String pageStr = sc.nextLine().trim();
        int page = pageStr.isEmpty() ? 1 : Integer.parseInt(pageStr);
        engine.search(query, page);
    }

    static void addDocumentMenu() {
        System.out.println("\n--- Add Document ---");
        System.out.print("URL   : "); String url     = sc.nextLine().trim();
        System.out.print("Title : "); String title   = sc.nextLine().trim();
        System.out.print("Content: "); String content = sc.nextLine().trim();
        engine.addDocument(url, title, content);
        System.out.println("✅ Document added and indexed!");
    }

    static void crawlMenu() {
        System.out.print("\nEnter URL to crawl: ");
        String url = sc.nextLine().trim();
        engine.crawl(url);
        System.out.println("Crawling started in background thread...");
        try { Thread.sleep(4000); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
    }

    static void loadSampleDocuments() {
        System.out.println("\nLoading sample documents...");
        engine.addDocument("https://example.com/java", "Java Programming Language",
                "Java is a high level object oriented programming language. Java is used for web development, mobile apps, and enterprise software. Java supports multithreading and is platform independent.");
        engine.addDocument("https://example.com/python", "Python Programming Language",
                "Python is a versatile high level programming language. Python is used for data science, machine learning, web development and automation. Python has simple and readable syntax.");
        engine.addDocument("https://example.com/dsa", "Data Structures and Algorithms",
                "Data structures are ways of organizing data. Algorithms are step by step procedures for solving problems. Common data structures include arrays, linked lists, trees, graphs, and hash maps.");
        engine.addDocument("https://example.com/ml", "Machine Learning Basics",
                "Machine learning is a subset of artificial intelligence. Algorithms learn from data to make predictions. Common algorithms include linear regression, decision tree, random forest and neural networks.");
        engine.addDocument("https://example.com/search", "How Search Engines Work",
                "Search engines use web crawlers to index pages. Inverted index stores word to document mappings. TF-IDF ranking algorithm scores documents based on term frequency and inverse document frequency.");
        System.out.println("✅ 5 sample documents loaded!\n");
    }
}