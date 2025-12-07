"""UI组件定义"""
import streamlit as st
from typing import Dict, Optional
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_config


def render_header():
    """渲染页面头部"""
    st.markdown("""
    <div class="main-header" role="banner" aria-label="NewsBoost Application Header">
        <h1 id="main-title">🌐 NewsBoost</h1>
        <p id="main-subtitle">Advanced News Analysis & Sentiment Intelligence</p>
        <span class="sr-only">NewsBoost is a comprehensive news analysis platform that provides sentiment analysis, keyword extraction, and data visualization for news articles.</span>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar() -> Dict:
    """渲染侧边栏并返回配置参数"""
    config = get_config()
    
    st.sidebar.markdown("### 🔧 Configuration")
    st.sidebar.markdown("---")
    
    # Search parameters
    with st.sidebar.expander("🔍 Search Parameters", expanded=True):
        query = st.text_input(
            "Search Query", 
            value="artificial intelligence", 
            help="Enter keywords to search for"
        )
        
        region = st.selectbox(
            "Region", 
            options=['US', 'UK', 'CA', 'AU', 'NG', 'IN', 'DE', 'FR'],
            help="Select geographical region"
        )
        
        category_map = {
            'General': None,
            'Business': 'BUSINESS',
            'Technology': 'TECHNOLOGY', 
            'Health': 'HEALTH',
            'Science': 'SCIENCE',
            'Sports': 'SPORTS'
        }
        
        category = st.selectbox("Category", options=list(category_map.keys()))
        max_articles = st.slider(
            "Max Articles", 
            min_value=10, 
            max_value=100, 
            value=config.app.default_max_articles
        )
    
    # Filtering options
    with st.sidebar.expander("🔍 Filtering Options", expanded=False):
        keyword_filter = st.text_input(
            "Filter by Keywords", 
            help="Only display headlines containing these words; use commas to separate filter keywords"
        )
        
        max_headlines = st.text_input(
            "Max Headlines", 
            value=50,
            help="Input the maximum headlines to show"
        )
    
    # Visualization options
    with st.sidebar.expander("🎨 Visualization Settings", expanded=False):
        exclude_words = st.text_area(
            "Exclude Words from WordCloud and Top Keywords display", 
            value="news, says, new, get, make, with, this",
            help="Comma-separated words to exclude"
        )
        
        colormap = st.selectbox(
            "WordCloud Color Scheme", 
            options=['modern', 'corporate', 'warm', 'viridis', 'plasma', 'inferno', 'magma', 'Blues'],
            help="Choose from modern, corporate, warm themes or classic matplotlib colormaps"
        )
    
    st.sidebar.markdown("---")
    
    return {
        'query': query,
        'region': region,
        'category': category_map[category],
        'max_articles': max_articles,
        'keyword_filter': keyword_filter,
        'max_headlines': max_headlines,
        'exclude_words': exclude_words,
        'colormap': colormap
    }


def render_footer():
    """渲染页面底部"""
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

