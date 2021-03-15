Input Files:
	dev_article : 
		- Text_ID : unique article ids
		- Text: Article Text Data
		- Headline: Headline to the article
		- Mobile_Tech_Flag: Flag shows whether article is related to mobile_tech 
	dev_tweet : 
		- Text_ID : unique tweet ids
		- Text: Tweet Text Data
		- Mobile_Tech_Flag: Flag shows whether tweet is related to mobile_tech

Output:
	sample_output_1: 
		- Text_ID : unique article/tweet ids
		- Mobile_Tech_Flag_Actual: Have actual mobile_tech values
		- Mobile_Tech_Flag_Predicted: Have predicted mobile_tech values
		- Headline_Actual_Eng: Headline to the article in English language
		- Headline_Generated_Eng_Lang: Generated headline to the article in English language

	sample_output_2: 
		- Text_ID : unique tweet ids
		- Mobile_Tech_Flag_Actual: Have actual mobile_tech values
		- Mobile_Tech_Flag_Predicted: Have predicted mobile_tech values
		- Brands_Entity_Actual: Actual Brands available in data
		- Sentiment_Actual: Actual Sentiment available in data	
		- Brands_Entity_Identified: Predicted Brands available in data	
		- Sentiment_Identified: Predicted Sentiment available in data	
