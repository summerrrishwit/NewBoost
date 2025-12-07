"""NewsBoost 主应用"""
import streamlit as st
import pandas as pd
from news_boost.utils import get_analyzers, process_sentiment_analysis, generate_keyword_analysis
from news_boost.exporter import DataExporter
from ui import apply_custom_styles, render_header, render_sidebar, render_footer
from ui.layout import (
    render_headlines_tab, 
    render_analytics_tab, 
    render_summary_tab, 
    render_data_tab, 
    render_export_tab
)
from config import get_config


def main():
    """主 Streamlit 应用"""
    config = get_config()
    
    # 页面配置
    st.set_page_config(
        page_title=config.app.page_title,
        page_icon=config.app.page_icon,
        layout=config.app.layout,
        initial_sidebar_state="expanded"
    )
    
    # 应用自定义样式
    apply_custom_styles()
    
    # 渲染头部
    render_header()
    
    # 初始化组件
    collector, analyzer, summarizer, visualizer = get_analyzers()
    
    # 渲染侧边栏并获取配置
    sidebar_config = render_sidebar()
    
    # 主内容
    if st.sidebar.button("🚀 Analyze News", type="primary"):
        # 同步收集新闻数据
        with st.spinner("Collecting news data..."):
            articles = collector.collect_news_data(
                query=sidebar_config['query'] if sidebar_config['query'] else None,
                region=sidebar_config['region'],
                category=sidebar_config['category'],
                max_articles=sidebar_config['max_articles']
            )
        
        if not articles:
            st.error("No articles found. Try adjusting your search parameters.")
            return
        
        # 转换为 DataFrame
        df = pd.DataFrame(articles)
        
        # 应用关键词过滤
        if sidebar_config['keyword_filter']:
            keywords = [k.strip().lower() for k in sidebar_config['keyword_filter'].split(',')]
            mask = df['title'].str.lower().str.contains('|'.join(keywords), na=False)
            df = df[mask]
        
        if df.empty:
            st.warning("No articles match your filter criteria.")
            return
        
        # 同步情感分析
        with st.spinner("Analyzing sentiment..."):
            titles = df['title'].tolist()
            sentiment_results = process_sentiment_analysis(titles)
            
            # 显示进度
            progress_bar = st.progress(0)
            for idx in range(len(titles)):
                progress_bar.progress((idx + 1) / len(titles))
            progress_bar.empty()
        
        # 添加情感数据到 DataFrame
        sentiment_df = pd.DataFrame(sentiment_results)
        df = pd.concat([df.reset_index(drop=True), sentiment_df.reset_index(drop=True)], axis=1)
        
        # 响应式指标显示
        st.subheader("📊 Analysis Metrics")
        
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
        
        # 首字母大写情感标签
        df['sentiment_label'] = df['sentiment_label'].str.capitalize()
        
        # 增强的标签页
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📰 Headlines", 
            "🎨 Analytics", 
            "📝 AI Summary", 
            "📋 Raw Data", 
            "💾 Export"
        ])
        
        # 生成词云（用于导出tab）
        exclude_words_set = {w.strip() for w in sidebar_config['exclude_words'].split(',') if w.strip()}
        wordcloud = visualizer.create_wordcloud(
            df['title'].tolist(), 
            exclude_words=exclude_words_set,
            colormap=sidebar_config['colormap']
        )
        
        with tab1:
            render_headlines_tab(df, sidebar_config['max_headlines'], sidebar_config['exclude_words'])
        
        with tab2:
            render_analytics_tab(df, visualizer, sidebar_config['exclude_words'], sidebar_config['colormap'])
        
        with tab3:
            render_summary_tab(df, summarizer, sidebar_config['exclude_words'])
        
        with tab4:
            render_data_tab(df)
        
        with tab5:
            render_export_tab(df, wordcloud)
    
    # 渲染底部
    render_footer()


if __name__ == "__main__":
    main()
