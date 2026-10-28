# Dataset folder

Place the cleaned dataset from the SI notebook here as:

    data/bigbasket_cleaned.csv

The notebook (section 10, "Save cleaned data") writes this file. The app finds it
automatically. If the folder is empty when you start the app, it shows a one-time
upload screen that saves the file here for you.

Accepted column names
- Cleaned names: Item_ID, Product, Category, Subcategory, Brand, Price, MRP,
  Product_Type, Rating, Price_Segment, Discount
- The notebook's raw names: index_items, product_name, category_name, sub_category,
  brand_name, sale_price, market_price, type_product, rating_product, price_segment
- The public Kaggle file "BigBasket Products.csv" (index, product, category,
  sub_category, brand, sale_price, market_price, type, rating, description)

Required: Product, Category, Subcategory, Brand, Price, MRP, Product_Type.
Discount is always recalculated from MRP and Price.
