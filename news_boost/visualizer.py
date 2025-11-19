import streamlit as st
import re
from collections import Counter

# Visualization libraries
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.graph_objects as go

# Global color function for wordcloud to avoid pickle issues
def create_color_func(color_scheme):
	"""Create a color function for wordcloud that can be pickled"""
	if isinstance(color_scheme, list):
		def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
			return color_scheme[random_state.randint(0, len(color_scheme) - 1)]
		return color_func
	return None

class DataVisualizer:
	"""Advanced data visualization for insights"""
	
	def __init__(self):
		plt.style.use('seaborn-v0_8')

	def create_wordcloud(self, text_data, exclude_words, colormap='viridis'):
		"""Generate enhanced customizable word cloud with modern styling"""
		
		# Convert set to sorted list for consistent processing
		exclude_words = sorted(list(exclude_words)) if exclude_words else [] 
		
		# Clean and combine text
		text = ' '.join(text_data).lower()
		text = re.sub(r'[^\w\s-]', ' ', text)
		
		# Remove common words
		words = [word for word in text.split() if len(word) > 3 and word not in exclude_words]
		text = ' '.join(words)
		
		if not text.strip():
			return None
		
		# Enhanced color schemes
		color_schemes = {
			'viridis': 'viridis',
			'plasma': 'plasma', 
			'inferno': 'inferno',
			'magma': 'magma',
			'Blues': 'Blues',
			'modern': ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe'],
			'corporate': ['#1e3a8a', '#3b82f6', '#60a5fa', '#93c5fd', '#dbeafe'],
			'warm': ['#dc2626', '#ea580c', '#d97706', '#ca8a04', '#65a30d']
		}
		
		# Select color scheme
		if colormap in color_schemes:
			color_scheme = color_schemes[colormap]
		else:
			color_scheme = 'viridis'
		
		# Create color function for custom color schemes
		color_func = create_color_func(color_scheme)
			
		wordcloud = WordCloud(
			width=1000, 
			height=500, 
			background_color='white',
			colormap=color_scheme if isinstance(color_scheme, str) else None,
			color_func=color_func,
			max_words=150,
			relative_scaling=0.3,
			random_state=42,
			regexp=r'\b(?!\d+\b)[\w-]+\b',  # Allow hyphens, strip pure numbers
			font_path=None,  # Use default font
			prefer_horizontal=0.9,
			mask=None,
			contour_width=0,
			contour_color='white',
			scale=2,
			min_font_size=10,
			max_font_size=200,
			font_step=1,
			mode='RGB',
			repeat=False,
			include_numbers=False,
			min_word_length=3,
			collocations=True,
			normalize_plurals=True
		).generate(text)
		
		return wordcloud
		
	@st.cache_data
	def plot_sentiment_distribution(_self, sentiments):
		"""Create enhanced sentiment distribution chart with modern styling"""
		
		# Define the fixed sentiment order and corresponding colors
		sentiment_order = ['Positive', 'Neutral', 'Negative']
		color_map = {
			'Positive': '#10b981',  # Modern green
			'Neutral': '#6b7280',   # Modern gray
			'Negative': '#ef4444'   # Modern red
		}
		
		# Count sentiments in the data
		sentiment_counts = Counter(sentiments)
		
		# Build y-values based on fixed order, using 0 if a sentiment is missing
		y_values = [sentiment_counts.get(sentiment, 0) for sentiment in sentiment_order]
		colors = [color_map[sentiment] for sentiment in sentiment_order]
		
		# Calculate percentages for display
		total = sum(y_values)
		percentages = [(count/total*100) if total > 0 else 0 for count in y_values]
		
		# Create enhanced bar chart
		fig = go.Figure(data=[
			go.Bar(
				x=sentiment_order,
				y=y_values,
				marker_color=colors,
				marker_line=dict(color='white', width=2),
				text=[f'{count}<br>({pct:.1f}%)' for count, pct in zip(y_values, percentages)],
				textposition='auto',
				textfont=dict(size=14, color='white', family='Inter'),
				hovertemplate='<b>%{x}</b><br>Count: %{y}<br>Percentage: %{customdata:.1f}%<extra></extra>',
				customdata=percentages
			)
		])
		
		# Enhanced layout with modern styling
		fig.update_layout(
			title=dict(
				text="📊 Sentiment Distribution Analysis",
				font=dict(size=24, family='Inter', color='#1f2937'),
				x=0.5,
				xanchor='center'
			),
			xaxis=dict(
				title=dict(text="Sentiment Category", font=dict(size=16, family='Inter', color='#374151')),
				tickfont=dict(size=14, family='Inter', color='#6b7280'),
				gridcolor='#f3f4f6',
				linecolor='#e5e7eb'
			),
			yaxis=dict(
				title=dict(text="Article Count", font=dict(size=16, family='Inter', color='#374151')),
				tickfont=dict(size=14, family='Inter', color='#6b7280'),
				gridcolor='#f3f4f6',
				linecolor='#e5e7eb'
			),
			plot_bgcolor='white',
			paper_bgcolor='white',
			showlegend=False,
			margin=dict(l=60, r=60, t=80, b=60),
			height=500
		)
		
		return fig