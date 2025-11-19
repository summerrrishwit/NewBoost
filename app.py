import streamlit as st
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from news_boost.utils import get_analyzers, process_sentiment_analysis, generate_keyword_analysis
from news_boost.exporter import DataExporter

# Main function for the NewsBoost application.
# Sets up the Streamlit interface, collects and filters news articles,
# performs sentiment and keyword analysis, generates visualizations,
# and provides data export options.

def main():
	"""Main Streamlit application"""
	
	# Page configuration
	st.set_page_config(
		page_title="NewsBoost",
		page_icon="🌐",
		layout="wide",
		initial_sidebar_state="expanded"
	)
	
	# Modern Styling
	st.markdown("""
	<style>
	/* Import Google Fonts */
	@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
	
	/* Global Styles */
	* {
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
	}
	
	/* Main Header */
	.main-header {
		text-align: center;
		padding: 3rem 2rem;
		background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
		color: white;
		margin-bottom: 2rem;
		border-radius: 20px;
		box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
		transition: all 0.3s ease;
		position: relative;
		overflow: hidden;
	}
	
	.main-header::before {
		content: '';
		position: absolute;
		top: 0;
		left: 0;
		right: 0;
		bottom: 0;
		background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, transparent 50%, rgba(255,255,255,0.1) 100%);
		animation: shimmer 3s infinite;
	}
	
	@keyframes shimmer {
		0% { transform: translateX(-100%); }
		100% { transform: translateX(100%); }
	}
	
	.main-header:hover {
		transform: translateY(-2px);
		box-shadow: 0 25px 50px rgba(102, 126, 234, 0.4);
	}
	
	.main-header h1 {
		font-size: 3rem;
		font-weight: 700;
		margin: 0;
		text-shadow: 0 2px 4px rgba(0,0,0,0.3);
		position: relative;
		z-index: 1;
	}
	
	.main-header p {
		font-size: 1.2rem;
		font-weight: 400;
		margin: 0.5rem 0 0 0;
		opacity: 0.9;
		position: relative;
		z-index: 1;
	}
	
	/* Metric Cards */
	.metric-card {
		background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
		padding: 1.5rem;
		border-radius: 16px;
		border: 1px solid #e9ecef;
		color: #2d3748;
		transition: all 0.3s ease;
		cursor: pointer;
		box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
		margin-bottom: 1rem;
		position: relative;
		overflow: hidden;
	}
	
	.metric-card::before {
		content: '';
		position: absolute;
		top: 0;
		left: 0;
		width: 4px;
		height: 100%;
		background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
		transition: width 0.3s ease;
	}
	
	.metric-card:hover {
		transform: translateY(-4px);
		box-shadow: 0 12px 24px rgba(102, 126, 234, 0.15);
		border-color: #667eea;
	}
	
	.metric-card:hover::before {
		width: 6px;
	}
	
	.metric-card h4 {
		font-size: 1.1rem;
		font-weight: 600;
		margin: 0 0 0.75rem 0;
		line-height: 1.4;
		color: #1a202c;
	}
	
	.metric-card p {
		font-size: 0.9rem;
		margin: 0;
		color: #4a5568;
		line-height: 1.5;
	}
	
	.source-text {
		font-weight: 500;
		color: #667eea;
		transition: color 0.3s ease;
	}
	
	.metric-card:hover .source-text {
		color: #5a67d8;
		text-decoration: underline;
	}
	
	/* Sidebar Styling */
	.css-1d391kg {
		background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
	}
	
	/* Button Styling */
	.stButton > button {
		background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
		color: white;
		border: none;
		border-radius: 12px;
		padding: 0.75rem 2rem;
		font-weight: 600;
		transition: all 0.3s ease;
		box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
	}
	
	.stButton > button:hover {
		transform: translateY(-2px);
		box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
	}
	
	/* Tab Styling */
	.stTabs [data-baseweb="tab-list"] {
		gap: 8px;
	}
	
	.stTabs [data-baseweb="tab"] {
		background: #f8f9fa;
		border-radius: 12px;
		padding: 0.75rem 1.5rem;
		font-weight: 500;
		transition: all 0.3s ease;
	}
	
	.stTabs [aria-selected="true"] {
		background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
		color: white;
	}
	
	/* Metric Display */
	[data-testid="metric-container"] {
		background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
		border: 1px solid #e9ecef;
		border-radius: 12px;
		padding: 1rem;
		box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
		transition: all 0.3s ease;
	}
	
	[data-testid="metric-container"]:hover {
		transform: translateY(-2px);
		box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
	}
	
	/* Progress Bar */
	.stProgress > div > div > div > div {
		background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
	}
	
	/* Footer */
	.footer {
		text-align: center;
		padding: 2rem;
		background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
		border-radius: 16px;
		margin-top: 2rem;
		color: #6c757d;
	}
	
	/* Responsive Design */
	@media (max-width: 768px) {
		.main-header h1 {
			font-size: 2rem;
		}
		
		.main-header p {
			font-size: 1rem;
		}
		
		.metric-card {
			padding: 1rem;
		}
	}
	
	/* Loading Animation */
	@keyframes pulse {
		0%, 100% { opacity: 1; }
		50% { opacity: 0.5; }
	}
	
	.pulse {
		animation: pulse 2s infinite;
	}
	
	/* Custom Scrollbar */
	::-webkit-scrollbar {
		width: 8px;
	}
	
	::-webkit-scrollbar-track {
		background: #f1f1f1;
		border-radius: 4px;
	}
	
	::-webkit-scrollbar-thumb {
		background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
		border-radius: 4px;
	}
	
	::-webkit-scrollbar-thumb:hover {
		background: linear-gradient(180deg, #5a67d8 0%, #6b46c1 100%);
	}
	
	/* Accessibility Improvements */
	.metric-card:focus {
		outline: 3px solid #667eea;
		outline-offset: 2px;
	}
	
	.metric-card[tabindex="0"]:focus {
		box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.3);
	}
	
	/* Screen reader only text */
	.sr-only {
		position: absolute;
		width: 1px;
		height: 1px;
		padding: 0;
		margin: -1px;
		overflow: hidden;
		clip: rect(0, 0, 0, 0);
		white-space: nowrap;
		border: 0;
	}
	
	/* High contrast mode support */
	@media (prefers-contrast: high) {
		.metric-card {
			border: 2px solid #000;
		}
		
		.main-header {
			background: #000;
			color: #fff;
		}
	}
	
	/* Reduced motion support */
	@media (prefers-reduced-motion: reduce) {
		* {
			animation-duration: 0.01ms !important;
			animation-iteration-count: 1 !important;
			transition-duration: 0.01ms !important;
		}
		
		.main-header::before {
			animation: none;
		}
	}
	
	/* Focus indicators for keyboard navigation */
	.stButton > button:focus,
	.stSelectbox > div:focus,
	.stTextInput > div:focus,
	.stTextArea > div:focus {
		outline: 3px solid #667eea;
		outline-offset: 2px;
	}
	</style>
	""", unsafe_allow_html=True)
	
	# Header with accessibility attributes
	st.markdown("""
	<div class="main-header" role="banner" aria-label="NewsBoost Application Header">
		<h1 id="main-title">🌐 NewsBoost</h1>
		<p id="main-subtitle">Advanced News Analysis & Sentiment Intelligence</p>
		<span class="sr-only">NewsBoost is a comprehensive news analysis platform that provides sentiment analysis, keyword extraction, and data visualization for news articles.</span>
	</div>
	""", unsafe_allow_html=True)
	
	# Initialize components
	collector, analyzer, summarizer, visualizer = get_analyzers()
	
	# Enhanced Sidebar Configuration
	st.sidebar.markdown("### 🔧 Configuration")
	st.sidebar.markdown("---")
	
	# Search parameters with better organization
	with st.sidebar.expander("🔍 Search Parameters", expanded=True):
		query = st.text_input("Search Query", value="artificial intelligence", 
							 help="Enter keywords to search for")
		
		region = st.selectbox("Region", 
							 options=['US', 'UK', 'CA', 'AU', 'NG', 'IN', 'DE', 'FR'],
							 help="Select geographical region")
		
		category_map = {
			'General': None,
			'Business': 'BUSINESS',
			'Technology': 'TECHNOLOGY', 
			'Health': 'HEALTH',
			'Science': 'SCIENCE',
			'Sports': 'SPORTS'
		}
		
		category = st.selectbox("Category", options=list(category_map.keys()))
		
		max_articles = st.slider("Max Articles", min_value=10, max_value=100, value=50)
	
	# Filtering options
	with st.sidebar.expander("🔍 Filtering Options", expanded=False):
		keyword_filter = st.text_input("Filter by Keywords", 
									  help="Only display headlines containing these words; use commas to separate filter keywords")
		
		max_headlines = st.text_input("Max Headlines", value=50,
									  help="Input the maximum headlines to show; if input is invalid, this will default to the total articles found if the total articles are at most 50, else this will default to 50 articles; actual headlines displayed may be fewer than your chosen value")
	
	# Visualization options
	with st.sidebar.expander("🎨 Visualization Settings", expanded=False):
		exclude_words = st.text_area("Exclude Words from WordCloud and Top Keywords display", 
									value="news, says, new, get, make, with, this",
									help="Comma-separated words to exclude")
		
		colormap = st.selectbox("WordCloud Color Scheme", 
							   options=['modern', 'corporate', 'warm', 'viridis', 'plasma', 'inferno', 'magma', 'Blues'],
							   help="Choose from modern, corporate, warm themes or classic matplotlib colormaps")
	
	st.sidebar.markdown("---")
	
	# Main content
	if st.sidebar.button("🚀 Analyze News", type="primary"):
		
		with st.spinner("Collecting news data..."):
			# Collect news data
			articles = collector.collect_news_data(
				query=query if query else None,
				region=region,
				category=category_map[category],
				max_articles=max_articles
			)
		
		if not articles:
			st.error("No articles found. Try adjusting your search parameters.")
			return
		
		# Convert to DataFrame
		df = pd.DataFrame(articles)
		
		# Apply keyword filtering
		if keyword_filter:
			keywords = [k.strip().lower() for k in keyword_filter.split(',')]
			mask = df['title'].str.lower().str.contains('|'.join(keywords), na=False)
			df = df[mask]
		
		if df.empty:
			st.warning("No articles match your filter criteria.")
			return
		
		# Perform sentiment analysis
		with st.spinner("Analyzing sentiment..."):
			titles = df['title'].tolist()
			sentiment_results = process_sentiment_analysis(titles)
	
			# Show visual progress
			progress_bar = st.progress(0)
			for idx in range(len(titles)):
				progress_bar.progress((idx + 1) / len(titles))
			progress_bar.empty()
		
		# Add sentiment data to DataFrame
		sentiment_df = pd.DataFrame(sentiment_results)
		df = pd.concat([df.reset_index(drop=True), sentiment_df.reset_index(drop=True)], axis=1)
		
		# Responsive metrics display
		st.subheader("📊 Analysis Metrics")
		
		# Use responsive columns based on screen size
		col1, col2, col3 = st.columns(3)
		col4, col5, col6 = st.columns(3)
		
		sentiment_counts = df['sentiment_label'].value_counts()
		total_articles = len(df)

		with col1:
			st.metric("Total Articles", total_articles)
		
		with col2:
			positive_pct = (sentiment_counts.get('positive', 0) / total_articles) * 100
			st.metric("Positive %", f"{positive_pct:.1f}%")

		with col3:
			neutral_pct = (sentiment_counts.get('neutral', 0) / total_articles) * 100
			st.metric("Neutral %", f"{neutral_pct:.1f}%")

		with col4:
			negative_pct = (sentiment_counts.get('negative', 0) / total_articles) * 100
			st.metric("Negative %", f"{negative_pct:.1f}%")
		
		if negative_pct > 0:
			with col5:
				ratio = positive_pct / negative_pct
				st.metric("Positive-to-Negative Ratio", f"{ratio:.2f}")
			
		with col6:
			top_source = df['source'].value_counts().index[0] if total_articles > 0 else "N/A"
			st.metric("Top Source", top_source)
		
		# Capitalize the first letter of each sentiment label (e.g., 'positive' → 'Positive') before proceeding
		df['sentiment_label'] = df['sentiment_label'].str.capitalize()
		
		# Enhanced tabs with better organization
		tab1, tab2, tab3, tab4, tab5 = st.tabs([
			"📰 Headlines", 
			"🎨 Analytics", 
			"📝 AI Summary", 
			"📋 Raw Data", 
			"💾 Export"
		])
		
		with tab1:
			# Convert input to integer if it's a digit-only string (e.g., "50"); otherwise, use total_articles as fallback
			max_headlines = int(max_headlines.strip()) if max_headlines.strip().isdigit() else total_articles
			
			st.header("📰 News Headlines")
			st.markdown("*Click on any headline to read the full article*")
			
			# Show sentiment statistics
			sentiment_counts = df['sentiment_label'].value_counts()
			col1, col2, col3, col4 = st.columns(4)
			
			with col1:
				positive_count = sentiment_counts.get('Positive', 0)
				st.metric("😊 Positive", positive_count)
			with col2:
				neutral_count = sentiment_counts.get('Neutral', 0)
				st.metric("😐 Neutral", neutral_count)
			with col3:
				negative_count = sentiment_counts.get('Negative', 0)
				st.metric("😞 Negative", negative_count)
			with col4:
				total_count = len(df)
				st.metric("📰 Total", total_count)
			
			st.markdown("---")
			
			# Add sentiment filter
			sentiment_filter = st.selectbox(
				"Filter by Sentiment", 
				options=["All", "Positive", "Neutral", "Negative"],
				key="sentiment_filter"
			)
			
			# Filter data based on sentiment
			if sentiment_filter == "All":
				filtered_df = df
			else:
				# Direct matching since sentiment_label is already capitalized
				filtered_df = df[df['sentiment_label'] == sentiment_filter]
			
			# Update displayable headlines based on filtered data
			displayable_headlines = min(max_headlines, len(filtered_df))
			
			# Show filter results with better formatting
			if sentiment_filter != "All":
				if len(filtered_df) == 0:
					st.warning(f"No {sentiment_filter.lower()} articles found. Try selecting a different sentiment filter.")
				else:
					st.success(f"Found {len(filtered_df)} {sentiment_filter.lower()} articles. Showing {displayable_headlines} of them.")
			else:
				st.info(f"Showing {displayable_headlines} articles out of {len(df)} total articles")
			
			# Display headlines with enhanced styling
			for idx, (_, row) in enumerate(filtered_df.iterrows()):
				if idx >= displayable_headlines:
					break
					
				sentiment_color = {
					'Positive': '#10b981', 
					'Neutral': '#6b7280',
					'Negative': '#ef4444'
				}.get(row['sentiment_label'], '#6b7280')
				
				# Add sentiment emoji
				sentiment_emoji = {
					'Positive': '😊',
					'Neutral': '😐', 
					'Negative': '😞'
				}.get(row['sentiment_label'], '❓')
				
				st.markdown(f"""
				<a href="{row['link']}" target="_blank" style="text-decoration: none; color: inherit;" 
				   aria-label="Read full article: {row['title']} from {row['source']} with {row['sentiment_label']} sentiment">
					<div class="metric-card" role="article" tabindex="0" 
						 aria-labelledby="headline-{idx}" aria-describedby="article-info-{idx}">
						<h4 id="headline-{idx}">{row['title']}</h4>
						<p id="article-info-{idx}"><strong>Source:</strong> 
							<span class="source-text">{row['source']}</span> | 
							<strong>Sentiment:</strong>
							<span style="color: {sentiment_color}; font-weight: 600;" 
								  aria-label="Sentiment: {row['sentiment_label']}">{sentiment_emoji} {row['sentiment_label']}</span>
						</p>
						<span class="sr-only">Click to read the full article in a new tab</span>
					</div>
				</a>
				""", unsafe_allow_html=True)
				st.markdown("<br>", unsafe_allow_html=True)
		
		with tab2:
			st.header("🎨 Analytics Dashboard")
			st.markdown("*Interactive visualizations and data insights*")
			
			# Create two columns for better layout
			col1, col2 = st.columns([2, 1])
			
			with col1:
				# Sentiment distribution
				st.subheader("📊 Sentiment Analysis")
				fig_sentiment = visualizer.plot_sentiment_distribution(df['sentiment_label'])
				st.plotly_chart(fig_sentiment, use_container_width=True)
			
			with col2:
				# Quick stats
				st.subheader("📈 Quick Stats")
				
				# Sentiment breakdown
				sentiment_breakdown = df['sentiment_label'].value_counts()
				for sentiment, count in sentiment_breakdown.items():
					percentage = (count / len(df)) * 100
					st.metric(
						label=f"{sentiment} Articles",
						value=count,
						delta=f"{percentage:.1f}%"
					)
				
				# Top sources
				st.subheader("📰 Top Sources")
				top_sources = df['source'].value_counts().head(3)
				for source, count in top_sources.items():
					st.write(f"• **{source}**: {count} articles")
			
			# Enhanced Word Cloud Section
			st.subheader("☁️ Word Cloud Analysis")
			st.markdown("*Visual representation of the most frequently mentioned terms in news headlines*")
			
			exclude_words = {w.strip() for w in exclude_words.split(',') if w.strip()}
			wordcloud = visualizer.create_wordcloud(df['title'].tolist(), 
												   exclude_words=exclude_words,
												   colormap=colormap)
			
			if wordcloud:
				# Create a container for the word cloud with modern styling
				col1, col2, col3 = st.columns([1, 2, 1])
				with col2:
					fig, ax = plt.subplots(figsize=(14, 8))
					ax.imshow(wordcloud, interpolation='bilinear')
					ax.axis('off')
					
					# Add subtle styling
					fig.patch.set_facecolor('white')
					fig.patch.set_alpha(0.0)
					
					st.pyplot(fig, use_container_width=True)
					
					# Add word cloud info
					st.info(f"📊 **Word Cloud Statistics:** {len(wordcloud.words_)} unique words displayed | Color scheme: {colormap}")
			else:
				st.warning("⚠️ Could not generate word cloud - insufficient text data or all words filtered out")
		
		with tab3:
			st.header("🤖 AI-Powered Analysis")
			st.markdown("*Intelligent insights and automated summarization*")
			
			# AI Summary Section
			col1, col2 = st.columns([2, 1])
			
			with col1:
				st.subheader("📝 AI Summary")
				if summarizer.available:
					with st.spinner("🤖 AI is analyzing headlines and generating insights..."):
						summary = summarizer.summarize_headlines(df['title'].tolist())
						st.success("✅ AI Analysis Complete!")
						st.markdown(f"""
						<div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
									padding: 1.5rem; border-radius: 12px; border-left: 4px solid #667eea;">
							<p style="font-size: 1.1rem; line-height: 1.6; margin: 0; color: #2d3748;">
								{summary}
							</p>
						</div>
						""", unsafe_allow_html=True)
				else:
					st.warning("⚠️ AI Summary feature not available - using fallback analysis")
					st.info("AI-powered summarization requires additional model downloads. Using keyword analysis instead.")
			
			with col2:
				st.subheader("⚡ Quick Insights")
				
				# Sentiment overview
				positive_count = len(df[df['sentiment_label'] == 'Positive'])
				negative_count = len(df[df['sentiment_label'] == 'Negative'])
				neutral_count = len(df[df['sentiment_label'] == 'Neutral'])
				
				if positive_count > negative_count:
					overall_sentiment = "😊 Generally Positive"
					sentiment_color = "#10b981"
				elif negative_count > positive_count:
					overall_sentiment = "😞 Generally Negative"
					sentiment_color = "#ef4444"
				else:
					overall_sentiment = "😐 Mixed/Neutral"
					sentiment_color = "#6b7280"
				
				st.metric("Overall Sentiment", overall_sentiment)
				
				# Top trending topics
				st.subheader("🔥 Trending Topics")
				top_keywords = generate_keyword_analysis(df['title'].tolist(), exclude_words)
				for i, (keyword, freq) in enumerate(top_keywords[:5]):
					st.write(f"{i+1}. **{keyword}** ({freq})")
			
			# Enhanced Keywords Analysis
			st.subheader("🔍 Keyword Analysis")
			top_keywords = generate_keyword_analysis(df['title'].tolist(), exclude_words)
			
			# Create a more visual keyword display
			keyword_df = pd.DataFrame(top_keywords, columns=['Keyword', 'Frequency'])
			
			# Add color coding based on frequency using a more reliable method
			def highlight_frequency(series):
				"""Apply color coding to frequency column"""
				colors = []
				for val in series:
					if val > series.quantile(0.8):
						colors.append('background-color: #d4edda; color: #155724;')
					elif val > series.quantile(0.6):
						colors.append('background-color: #fff3cd; color: #856404;')
					else:
						colors.append('background-color: #f8d7da; color: #721c24;')
				return colors
			
			# Apply styling to the dataframe
			styled_df = keyword_df.style.apply(highlight_frequency, subset=['Frequency'])
			st.dataframe(
				styled_df,
				use_container_width=True,
				height=400
			)
		
		with tab4:
			st.header("📋 Data Explorer")
			st.markdown("*Explore and filter the raw news data*")
			
			# Data overview
			col1, col2, col3, col4 = st.columns(4)
			with col1:
				st.metric("Total Articles", len(df))
			with col2:
				st.metric("Unique Sources", df['source'].nunique())
			with col3:
				if 'published' in df.columns and not df['published'].isna().all():
					try:
						# Try to convert to datetime if it's not already
						published_series = pd.to_datetime(df['published'], errors='coerce')
						min_date = published_series.min()
						if pd.notna(min_date):
							date_range = min_date.strftime('%Y-%m-%d')
						else:
							date_range = "N/A"
					except (TypeError, ValueError, AttributeError):
						# If conversion fails, just show the raw min value
						min_val = df['published'].min()
						date_range = str(min_val) if pd.notna(min_val) else "N/A"
				else:
					date_range = "N/A"
				st.metric("Date Range", date_range)
			with col4:
				st.metric("Avg Title Length", f"{df['title'].str.len().mean():.0f} chars")
			
			# Display options
			st.subheader("🔧 Data Filters")
			
			col1, col2 = st.columns(2)
			with col1:
				show_columns = st.multiselect(
					"Select columns to display",
					options=df.columns.tolist(),
					default=['title', 'source', 'sentiment_label'],
					help="Choose which columns to display in the data table"
				)
			
			with col2:
				# Add search functionality
				search_term = st.text_input("Search in titles", placeholder="Enter keywords to search...")
			
			# Filter data based on search
			display_df = df.copy()
			if search_term:
				display_df = display_df[display_df['title'].str.contains(search_term, case=False, na=False)]
				st.info(f"Found {len(display_df)} articles matching '{search_term}'")
			
			if show_columns:
				st.subheader("📊 Data Table")
				st.dataframe(
					display_df[show_columns], 
					use_container_width=True,
					height=500
				)
				
				# Add download option for filtered data
				if search_term:
					csv_data = DataExporter.to_csv(display_df[show_columns])
					st.download_button(
						label="📥 Download Filtered Data (CSV)",
						data=csv_data,
						file_name=f"filtered_news_data_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv",
						mime="text/csv"
					)
			else:
				st.warning("Please select at least one column to display.")
		
		with tab5:
			st.header("💾 Export & Download")
			st.markdown("*Export your analysis results in multiple formats*")
			
			# Export options with enhanced styling
			file_name = f"newsboost_analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"
			
			# Create a more organized export section
			st.subheader("📊 Data Exports")
			
			col1, col2 = st.columns(2)
			
			with col1:
				# CSV Export
				st.markdown("### 📄 CSV Format")
				st.markdown("*Perfect for Excel, Google Sheets, and data analysis tools*")
				csv_data = DataExporter.to_csv(df)
				st.download_button(
					label="📥 Download CSV",
					data=csv_data,
					file_name=f"{file_name}.csv",
					mime="text/csv",
					help="Download all data in CSV format for spreadsheet applications"
				)
			
			with col2:
				# JSON Export
				st.markdown("### 📋 JSON Format")
				st.markdown("*Ideal for APIs, databases, and programmatic access*")
				json_data = DataExporter.to_json(df)
				st.download_button(
					label="📥 Download JSON",
					data=json_data,
					file_name=f"{file_name}.json",
					mime="application/json",
					help="Download data in JSON format for integration with other applications"
				)
			
			# Visual exports
			st.subheader("🎨 Visual Exports")
			
			col1, col2 = st.columns(2)
			
			with col1:
				# WordCloud PNG Export
				st.markdown("### ☁️ Word Cloud Image")
				if 'wordcloud' in locals() and wordcloud:
					png_data = DataExporter.wordcloud_to_png(wordcloud)
					if png_data:
						st.download_button(
							label="🖼️ Download WordCloud PNG",
							data=png_data,
							file_name=f"{file_name}_wordcloud.png",
							mime="image/png",
							help="Download the word cloud as a high-resolution PNG image"
						)
					else:
						st.warning("Word cloud not available for download")
				else:
					st.info("Generate a word cloud first to enable download")
			
			with col2:
				# Analysis Summary Export
				st.markdown("### 📝 Analysis Summary")
				summary_text = f"""
NewsBoost Analysis Summary
Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC

Dataset Overview:
- Total Articles: {len(df)}
- Unique Sources: {df['source'].nunique()}
- Sentiment Distribution:
  * Positive: {len(df[df['sentiment_label'] == 'Positive'])} articles
  * Neutral: {len(df[df['sentiment_label'] == 'Neutral'])} articles  
  * Negative: {len(df[df['sentiment_label'] == 'Negative'])} articles

Top Sources:
{chr(10).join([f"- {source}: {count} articles" for source, count in df['source'].value_counts().head(5).items()])}

Analysis completed using NewsBoost - Advanced News Analysis Platform
"""
				st.download_button(
					label="📄 Download Summary",
					data=summary_text,
					file_name=f"{file_name}_summary.txt",
					mime="text/plain",
					help="Download a text summary of the analysis results"
				)
			
			# Export statistics
			st.subheader("📈 Export Statistics")
			col1, col2, col3 = st.columns(3)
			with col1:
				st.metric("Total Records", len(df))
			with col2:
				st.metric("File Size (CSV)", f"{len(csv_data.encode('utf-8')) / 1024:.1f} KB")
			with col3:
				st.metric("Export Time", f"{datetime.utcnow().strftime('%H:%M:%S')}")
	
	# Footer
	st.markdown("---")
	st.markdown("""
	<div class="footer">
		<p style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; color: #495057;">
			🌐 NewsBoost - Built for tracking media narratives in real time
		</p>
		<p style="font-size: 0.9rem; margin: 0; opacity: 0.8;">
			Powered by Advanced NLP & Real-time Data Analysis
		</p>
	</div>
	""", unsafe_allow_html=True)

if __name__ == "__main__":
	main()