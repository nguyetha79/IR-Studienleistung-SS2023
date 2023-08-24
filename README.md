# IR-SL1-Indexieren-Suchen
---
### Die Dokumentation folgt hier: 

#### 1. Eine Beschreibung (und Begründung) Ihrer angewandten Methodik für das Indexieren von Dokumenten, sowohl in Bezug auf Art des Preprocessings als auch technische Umsetzung.

Bevor wir mit der Bearbeitung der Aufgaben angefangen haben, haben wir die Voraussetzungen dafür geschaffen diese überhaupt bearbeiten zu können. Wir haben die Methoden zur Create Index Method, Index Method und Reindex Method geschrieben, die wir zur Bearbeitung der Aufgaben benötigen. 
-> Indexierung: Methode, um den Inhalt der Dokumente zu analysieren, zu strukturieren und in einer durchsuchbaren Form zu speichern. 
-> Preprocessing: Vorverarbeitungsschritte, um den Text in eine standardisierte Form zu bringen und irrelevante Informationen zu entfernen. 
Die Wahl der Methodik hängt von verschiedenen Faktoren ab, einschließlich der Größe des Dokumentensatzes (in unserem Fall: Signal AI 1 Million News Articles Dataset), der Art der Dokumente und den spezifischen Anforderungen der Suche. 

Wir haben, wie in der Aufgabe gefordert, den Code zur Bearbeitung in Python geschrieben. 
Zunächst haben wir ein ein Python-Skript geschrieben, das Funktionen zur Verfügung stellt, um Dokumente in Elasticsearch zu indexieren und zu verarbeiten. Dabei sind wir in den folgenden Schritten vorgegangen: 
- 	create_index(index_name, mappings): Diese Funktion erstellt einen Index in Elasticsearch mit dem angegebenen Indexnamen und den Mappings. Mappings definieren die Datenstruktur der 	indexierten Dokumente, einschließlich der Datentypen der Felder. Die Funktion erstellt den Index.
-	index_docs(document_dicts, index_url): Diese Funktion indexiert eine Liste von Dokumenten. Die Funktion verwendet eine Schleife, um jedes Dokument in der Liste zu durchlaufen und 	um das Dokument zu indexieren. Der index_url-Parameter definiert den Endpunkt, an dem das Dokument indexiert werden soll.
-	reindex(num_docs): Diese Funktion führt eine Neuindexierung der Dokumenten durch. Hiermit werden die Dokumente von einem vorhandenen Index in einen neuen Index zu kopieren. Der 	num_docs-Parameter gibt an, wie viele Dokumente neu indexiert werden sollen.
-	index_articles(num_docs): Diese Funktion führt den Indexierungsprozess für Artikel aus. Sie erstellt zunächst einen Index für die ursprünglichen Artikel-Daten mit dem angegebenen 	num_docs-Wert und indexiert die entsprechende Anzahl von Dokumenten aus der angegebenen JSONL-Datei. Anschließend wird ein neuer Index für die verarbeiteten Artikel-Daten erstellt, 	der Preprocessing-Schritte wie die Verwendung des English Analyzers für das Inhaltsfeld enthält. Wir haben uns für den English Analyzer entschieden, da der zu verarbeitende 	Datensatz nur englischsprachige Artikel enthält. Durch den English Analyzer kann speziell das analysiert werden. Mittels des Analyzers werden sprachspezifische Stopwörter entfernt 	und die verschiedenen Bearbeitungsschritte wie Stemming durchgeführt werden. So ist es möglich, dass der Text an Ende so aufgebreitet ist, dass er zur Analyse und Suche bereit ist 	und die Sprache auf Englisch festgelegt ist. Darauf folgend wird die Neuindexierung der Dokumenten durchgeführt, um die verarbeiteten Daten in den neuen Index zu kopieren.

Um den Datensatz zu verarbeiten, wird die Elasticsearch-Python-Bibliothek verwendet. Dies ist notwendig, um HTTP-Anfragen an die Elasticsearch-Schnittstelle zu senden und mit Elasticsearch zu interagieren. Die Elasticsearch-URL und der Dateipfad für die JSONL-Datei werden als Konstanten definiert, und die erforderlichen Datenstrukturen (Mappings) für die Indexierung werden als Python-Wörterbuchobjekte festgelegt.

#### 2. Eine Beschreibung Ihrer Funktionen zum Durchsuchen von Dokumenten.

Mit dieser Methode für die Aufgaben 2a) wird zunächst das Dokument durchsucht:
- 	search_optional_params(query, index_name, fields): Diese Methode führt eine Suchabfrage durch und gibt die Ergebnisse basierend auf optionalen (gewählten) Parametern zurück. 
Die Methode erhält hier drei Parameter:
-	query: Ein Objekt, das die Abfrageinformationen enthält
-	index_name: Der Name des Index, auf den die Suche angewendet wird
-	fields: Eine Liste von Feldern, nach denen gesucht werden soll
-> Die Methode initialisiert eine leere Liste namens "data", in das später die Daten gespeichert werden.
Es wird eine Anfrage an Elasticsearch gesendet, um die Suche durchzuführen. Dafür benötigt man "index_name" und die Konstante "ELASTICSEARCH_URL". Die Suchabfrage wird als JSON-Format übergeben. Auch die Antwort von Elasticsearch wird in JSON-Format zergliedert und in der Variablen "res" gespeichert. Als nächstes werden mittels einer Schleife, die über "res" ausgeführt wird, die gefundenen Treffer in "_source"-Objekt extrahiert und das jeweilige Schlüssel-Wert-Paare in "data" hinzugefügt. Durch eine Comprehension werden anschließend die Schlüssel-Wert-Paare aus "data" in der Variable "results" gespeichert und die gefundenen Ergebnisse werden in "results" ausgegeben. Danach kann man mit diesen Ergebnissen weiterarbeiten und diese auf die Suchanfragen anwenden.

In die 2b) haben wir die Methode "search_index(query, index_name)" angewendet. Diese Methode enthält zwei Parameter: 
-	query: Ein Objekt, das die Abfrageinformationen enthält
-	index_name: Der Name des Elasticsearch-Index, auf den die Suche angewendet wird
Auch hier wird wieder eine Anfrage an Elasticsearch gesendet, um die Suche durchzuführen. Wie auch bei der oben beschriebenen Methode wird hier durch "index_name" und der Konstante "ELASTICSEARCH_URL" eine URL erstellt. Wie auch bei obiger Methode wird die Suchabfrage als JSON-Format übergeben. Die Antwort von Elasticsearch wird in der Variable "reponse" gespeichert. Durch die Durchführung der Methode werden zwei Ausgaben gemacht: Zunächst wird "Query reponse: " ausgegeben und der Inhalt der Antwort "reponse" wird als JSON formatiert und ausgegeben.

Die Methode verwendet eine GET-Anfrage, um die Suche auszuführen. Die Parameter sind:
-	query: Die Suchabfrage
-	index_name: Der Name des Index, in dem die Suche durchgeführt werden soll
-	fields: Eine Liste von Feldern, die in den Suchergebnissen zurückgegeben werden sollen
-	search_index(query, index_name): Diese Methode führt eine Suchabfrage durch und gibt das abgefragten Ergebnisse zurück. Auch hier wird wieder eine GET-Anfrage verwendet, um die 	Suche auszuführen. Die Parameter sind dieselben wie bei search_optional_params.

Es wurden Suchabfragen für Aufgabe 2 und Aufgabe 3 (siehe unten) bereitgestellt. Die Abfragen sind als Python-Wörterbuchobjekte definiert und enthalten die entsprechenden Suchkriterien gemäß den Aufgabenbeschreibungen.

3. Eine Beschreibung der zehn Suchanfragen, die Sie sich überlegt haben.
Mit Hilfe des Nachrichtenarchivs der Süddeutschen Zeitung haben wir nach Nachrichtenthemen aus dem Monat September 2015 gesucht (Quelle: https://www.sueddeutsche.de/archiv/). Dabei haben wir auf relevante Themen mit internationaler Tragweite geachtet und Suchanfragen formuliert. Dabei sollten die Anfragen breitgefächert über die verschiedenen Themengebiete in der Nachrichtenwelt gestreut sein. Wir haben im Rahmen unserer Recherche daher auf regionale Themen verzichtet, auch Themen die nicht von Relevanz für das Weltgeschehen waren und Deutschland spezifisch waren, wurden nicht einbezogen.
Unsere 10 Suchanfragen sind: 
1.	Wer hat das Magazin „National Geographic“ gekauft? / Who bought National Geographic magazine?
2.	Vor was fliehen die Menschen in Syrien? / What are people fleeing from in Syria?
3.	Wie viel kostet das iPad Pro? / How much does the iPad Pro cost?
4.	Was wurde auf dem Mars entdeckt? / What was discovered on Mars?
5.	Wie viele Menschen fliehen nach Deutschland? / How many people flee to Germany?
6.	Welche Künstler sind die Headliner auf dem Apple Music Festival? / Which artists are the headliners at Apple Music Festival?
7.	Wann hat Edward Snowden in seinem ersten Twitter-Post geschrieben? / When did Edward Snowden write his first Twitter post?
8.	Welches Land hat sein erstes Weltraum-Observatorium ins Weltall geschossen? / Which country launched its first space observatory into space?
9.	Was fordert Kanzlerin Merkl von Facebook? / What is Chancellor Merkl asking of Facebook?
10.	Was führte zum Zusammenbruch von Facebook? / Why went Facebook down?

Da es sich bei dem Datensatz um einen englischsprachigen handelt, haben wir nur mit den auf englisch formulierten Fragestellungen gesucht. 

