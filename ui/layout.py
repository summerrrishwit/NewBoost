"""UI布局和Tab渲染函数"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from typing import Set
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from news_boost.utils import generate_keyword_analysis
from news_boost.exporter import DataExporter
from datetime import datetime


def render_headlines_tab(df: pd.DataFrame, max_headlines: str, exclude_words: str):
    """渲染新闻标题Tab"""
    total_articles = len(df)
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
        st.metric("📰 Total", total_articles)
    
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
        filtered_df = df[df['sentiment_label'] == sentiment_filter]
    
    # Update displayable headlines
    displayable_headlines = min(max_headlines, len(filtered_df))
    
    # Show filter results
    if sentiment_filter != "All":
        if len(filtered_df) == 0:
            st.warning(f"No {sentiment_filter.lower()} articles found. Try selecting a different sentiment filter.")
        else:
            st.success(f"Found {len(filtered_df)} {sentiment_filter.lower()} articles. Showing {displayable_headlines} of them.")
    else:
        st.info(f"Showing {displayable_headlines} articles out of {total_articles} total articles")
    
    # Display headlines
    for idx, (_, row) in enumerate(filtered_df.iterrows()):
        if idx >= displayable_headlines:
            break
            
        sentiment_color = {
            'Positive': '#10b981', 
            'Neutral': '#6b7280',
            'Negative': '#ef4444'
        }.get(row['sentiment_label'], '#6b7280')
        
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


def render_analytics_tab(df: pd.DataFrame, visualizer, exclude_words: str, colormap: str):
    """渲染分析仪表板Tab"""
    st.header("🎨 Analytics Dashboard")
    st.markdown("*Interactive visualizations and data insights*")
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Sentiment Analysis")
        fig_sentiment = visualizer.plot_sentiment_distribution(df['sentiment_label'])
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    with col2:
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
    
    # Word Cloud Section
    st.subheader("☁️ Word Cloud Analysis")
    st.markdown("*Visual representation of the most frequently mentioned terms in news headlines*")
    
    exclude_words_set = {w.strip() for w in exclude_words.split(',') if w.strip()}
    wordcloud = visualizer.create_wordcloud(
        df['title'].tolist(), 
        exclude_words=exclude_words_set,
        colormap=colormap
    )
    
    if wordcloud:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(0.0)
            
            st.pyplot(fig, use_container_width=True)
            
            st.info(f"📊 **Word Cloud Statistics:** {len(wordcloud.words_)} unique words displayed | Color scheme: {colormap}")
    else:
        st.warning("⚠️ Could not generate word cloud - insufficient text data or all words filtered out")


def render_summary_tab(df: pd.DataFrame, summarizer, exclude_words: str):
    """渲染AI摘要Tab"""
    st.header("🤖 AI-Powered Analysis")
    st.markdown("*Intelligent insights and automated summarization*")
    
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
        elif negative_count > positive_count:
            overall_sentiment = "😞 Generally Negative"
        else:
            overall_sentiment = "😐 Mixed/Neutral"
        
        st.metric("Overall Sentiment", overall_sentiment)
        
        # Top trending topics
        st.subheader("🔥 Trending Topics")
        exclude_words_set = {w.strip() for w in exclude_words.split(',') if w.strip()}
        top_keywords = generate_keyword_analysis(df['title'].tolist(), exclude_words_set)
        for i, (keyword, freq) in enumerate(top_keywords[:5]):
            st.write(f"{i+1}. **{keyword}** ({freq})")
    
    # Keywords Analysis
    st.subheader("🔍 Keyword Analysis")
    exclude_words_set = {w.strip() for w in exclude_words.split(',') if w.strip()}
    top_keywords = generate_keyword_analysis(df['title'].tolist(), exclude_words_set)
    
    keyword_df = pd.DataFrame(top_keywords, columns=['Keyword', 'Frequency'])
    
    def highlight_frequency(series):
        colors = []
        for val in series:
            if val > series.quantile(0.8):
                colors.append('background-color: #d4edda; color: #155724;')
            elif val > series.quantile(0.6):
                colors.append('background-color: #fff3cd; color: #856404;')
            else:
                colors.append('background-color: #f8d7da; color: #721c24;')
        return colors
    
    styled_df = keyword_df.style.apply(highlight_frequency, subset=['Frequency'])
    st.dataframe(styled_df, use_container_width=True, height=400)


def render_data_tab(df: pd.DataFrame):
    """渲染数据浏览Tab"""
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
                published_series = pd.to_datetime(df['published'], errors='coerce')
                min_date = published_series.min()
                if pd.notna(min_date):
                    date_range = min_date.strftime('%Y-%m-%d')
                else:
                    date_range = "N/A"
            except (TypeError, ValueError, AttributeError):
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
        search_term = st.text_input("Search in titles", placeholder="Enter keywords to search...")
    
    # Filter data
    display_df = df.copy()
    if search_term:
        display_df = display_df[display_df['title'].str.contains(search_term, case=False, na=False)]
        st.info(f"Found {len(display_df)} articles matching '{search_term}'")
    
    if show_columns:
        st.subheader("📊 Data Table")
        st.dataframe(display_df[show_columns], use_container_width=True, height=500)
        
        # Download option
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


def render_export_tab(df: pd.DataFrame, wordcloud=None):
    """渲染导出Tab"""
    st.header("💾 Export & Download")
    st.markdown("*Export your analysis results in multiple formats*")
    
    file_name = f"newsboost_analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"
    
    st.subheader("📊 Data Exports")
    
    col1, col2 = st.columns(2)
    
    with col1:
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
        st.markdown("### ☁️ Word Cloud Image")
        if wordcloud:
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

