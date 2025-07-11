import pandas as pd
def finance(profile_data):
    suggestions=[]
    profiles=[]
    for idx,row in profile_data.iterrows():
        segment = row["segment"]
        if segment=='Premium Spenders':
            profile="High gross income, high spending, likely less price-sensitive."
            sug="Launch premium membership programs with exclusive deals and Promote high-margin or luxury products."
        elif segment=='Budget Buyers':
            profile="Low to mid-income, very price-conscious."
            sug="Use discount-driven promotions or loyalty points and Send targeted SMS/email coupons on essential items."
        elif segment=='Occasional Shoppers':
            profile=" Low engagement, sporadic purchases."
            sug="Trigger retargeting ads after inactivity and Collect feedback to understand why visits are infrequent."
        elif segment=='Product Line Loyalists':
            profile="Strong preference for specific product categories."
            sug="Recommend complementary products from other lines  and Track their favorite product line for auto-refill or subscription offers."
        else:
            profile = "General customer."
            suggestion = "Explore more engagement opportunities."
            suggestions.append(sug)
            profiles.append(profile)
    return pd.DataFrame({"Cluster":profile_data.index,"Customers profile":profiles,"Suggested Financial strategy":suggestions})