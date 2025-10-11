# Polars Cheat Sheet (for Pandas Users)

A quick reference guide for transitioning from pandas to Polars.

## Table of Contents
- [Reading Data](#reading-data)
- [Basic DataFrame Operations](#basic-dataframe-operations)
- [Filtering Data](#filtering-data)
- [Transformations](#transformations)
- [Aggregations](#aggregations)
- [Joins](#joins)
- [Writing Data](#writing-data)

## Reading Data

| Operation | Pandas | Polars |
|-----------|--------|--------|
| Read CSV | `pd.read_csv('file.csv')` | `pl.read_csv('file.csv')` |
| Read Parquet | `pd.read_parquet('file.parquet')` | `pl.read_parquet('file.parquet')` |
| Read JSON | `pd.read_json('file.json')` | `pl.read_json('file.json')` |
| Read Excel | `pd.read_excel('file.xlsx')` | `pl.read_excel('file.xlsx')` |
| Lazy reading | N/A | `pl.scan_csv('file.csv')` |

```python
import polars as pl

# Eager reading (loads immediately)
df = pl.read_csv('data.csv')

# Lazy reading (optimized execution)
df_lazy = pl.scan_csv('data.csv')
result = df_lazy.collect()  # Execute the query
```

## Basic DataFrame Operations

| Operation | Pandas | Polars |
|-----------|--------|--------|
| View first rows | `df.head()` | `df.head()` |
| View last rows | `df.tail()` | `df.tail()` |
| Shape | `df.shape` | `df.shape` |
| Column names | `df.columns` | `df.columns` |
| Data types | `df.dtypes` | `df.dtypes` or `df.schema` |
| Info | `df.info()` | `df.describe()` |
| Select columns | `df[['col1', 'col2']]` | `df.select(['col1', 'col2'])` |
| Drop columns | `df.drop(['col1'], axis=1)` | `df.drop(['col1'])` |
| Rename columns | `df.rename(columns={'old': 'new'})` | `df.rename({'old': 'new'})` |

```python
# Polars examples
df.head(10)
df.select(['name', 'age'])
df.select(pl.col('name'), pl.col('age'))
df.drop('column_to_remove')
```

## Filtering Data

| Operation | Pandas | Polars |
|-----------|--------|--------|
| Single condition | `df[df['age'] > 25]` | `df.filter(pl.col('age') > 25)` |
| Multiple conditions (AND) | `df[(df['age'] > 25) & (df['city'] == 'NYC')]` | `df.filter((pl.col('age') > 25) & (pl.col('city') == 'NYC'))` |
| Multiple conditions (OR) | `df[(df['age'] > 25) \| (df['city'] == 'NYC')]` | `df.filter((pl.col('age') > 25) \| (pl.col('city') == 'NYC'))` |
| String contains | `df[df['name'].str.contains('John')]` | `df.filter(pl.col('name').str.contains('John'))` |
| NULL values | `df[df['col'].isna()]` | `df.filter(pl.col('col').is_null())` |
| NOT NULL values | `df[df['col'].notna()]` | `df.filter(pl.col('col').is_not_null())` |
| Value in list | `df[df['city'].isin(['NYC', 'LA'])]` | `df.filter(pl.col('city').is_in(['NYC', 'LA']))` |

```python
# Polars filtering examples
df.filter(pl.col('age') > 25)
df.filter((pl.col('age') > 25) & (pl.col('salary') > 50000))
df.filter(pl.col('name').str.starts_with('A'))
df.filter(pl.col('email').str.ends_with('@gmail.com'))
df.filter(~pl.col('status').is_in(['inactive', 'deleted']))
```

## Transformations

### Adding/Modifying Columns

| Operation | Pandas | Polars |
|-----------|--------|--------|
| Add new column | `df['new'] = df['col1'] + df['col2']` | `df.with_columns((pl.col('col1') + pl.col('col2')).alias('new'))` |
| Modify column | `df['col'] = df['col'] * 2` | `df.with_columns((pl.col('col') * 2).alias('col'))` |
| Multiple columns | `df.assign(a=..., b=...)` | `df.with_columns([pl.col('x').alias('a'), pl.col('y').alias('b')])` |

```python
# Polars examples
df.with_columns(
    (pl.col('price') * 1.1).alias('price_with_tax')
)

df.with_columns([
    (pl.col('first_name') + ' ' + pl.col('last_name')).alias('full_name'),
    (pl.col('salary') / 12).alias('monthly_salary')
])

# Conditional column
df.with_columns(
    pl.when(pl.col('age') >= 18)
      .then(pl.lit('adult'))
      .otherwise(pl.lit('minor'))
      .alias('age_group')
)
```

### String Operations

| Operation | Pandas | Polars |
|-----------|--------|--------|
| Lowercase | `df['col'].str.lower()` | `pl.col('col').str.to_lowercase()` |
| Uppercase | `df['col'].str.upper()` | `pl.col('col').str.to_uppercase()` |
| Strip whitespace | `df['col'].str.strip()` | `pl.col('col').str.strip_chars()` |
| Replace | `df['col'].str.replace('old', 'new')` | `pl.col('col').str.replace('old', 'new')` |
| Split | `df['col'].str.split(',')` | `pl.col('col').str.split(',')` |
| Length | `df['col'].str.len()` | `pl.col('col').str.len_chars()` |

```python
df.with_columns([
    pl.col('name').str.to_lowercase().alias('name_lower'),
    pl.col('email').str.strip_chars().alias('email_clean'),
    pl.col('tags').str.split(',').alias('tag_list')
])
```

### Date/Time Operations

| Operation | Pandas | Polars |
|-----------|--------|--------|
| Parse dates | `pd.to_datetime(df['date'])` | `pl.col('date').str.strptime(pl.Date, '%Y-%m-%d')` |
| Extract year | `df['date'].dt.year` | `pl.col('date').dt.year()` |
| Extract month | `df['date'].dt.month` | `pl.col('date').dt.month()` |
| Extract day | `df['date'].dt.day` | `pl.col('date').dt.day()` |
| Date difference | `df['date2'] - df['date1']` | `pl.col('date2') - pl.col('date1')` |

```python
df.with_columns([
    pl.col('date_str').str.strptime(pl.Date, '%Y-%m-%d').alias('date'),
    pl.col('timestamp').dt.year().alias('year'),
    pl.col('timestamp').dt.month().alias('month'),
    pl.col('timestamp').dt.weekday().alias('day_of_week')
])
```

### Sorting

| Operation | Pandas | Polars |
|-----------|--------|--------|
| Sort by column | `df.sort_values('col')` | `df.sort('col')` |
| Sort descending | `df.sort_values('col', ascending=False)` | `df.sort('col', descending=True)` |
| Sort by multiple | `df.sort_values(['col1', 'col2'])` | `df.sort(['col1', 'col2'])` |

```python
df.sort('age')
df.sort('salary', descending=True)
df.sort(['department', 'salary'], descending=[False, True])
```

## Aggregations

| Operation | Pandas | Polars |
|-----------|--------|--------|
| Sum | `df['col'].sum()` | `df['col'].sum()` or `df.select(pl.col('col').sum())` |
| Mean | `df['col'].mean()` | `df['col'].mean()` or `df.select(pl.col('col').mean())` |
| Count | `df['col'].count()` | `df['col'].count()` or `df.select(pl.col('col').count())` |
| Min/Max | `df['col'].min()`, `df['col'].max()` | `df.select(pl.col('col').min())`, `df.select(pl.col('col').max())` |
| Unique | `df['col'].unique()` | `df['col'].unique()` |
| Value counts | `df['col'].value_counts()` | `df['col'].value_counts()` |

### GroupBy Operations

| Operation | Pandas | Polars |
|-----------|--------|--------|
| Group by | `df.groupby('col')['val'].sum()` | `df.group_by('col').agg(pl.col('val').sum())` |
| Multiple aggs | `df.groupby('col').agg({'a': 'sum', 'b': 'mean'})` | `df.group_by('col').agg([pl.col('a').sum(), pl.col('b').mean()])` |

```python
# Polars groupby examples
df.group_by('department').agg([
    pl.col('salary').mean().alias('avg_salary'),
    pl.col('salary').max().alias('max_salary'),
    pl.col('employee_id').count().alias('employee_count')
])

# Multiple groupby columns
df.group_by(['department', 'location']).agg([
    pl.col('sales').sum().alias('total_sales')
])
```

## Joins

| Operation | Pandas | Polars |
|-----------|--------|--------|
| Inner join | `df1.merge(df2, on='key')` | `df1.join(df2, on='key', how='inner')` |
| Left join | `df1.merge(df2, on='key', how='left')` | `df1.join(df2, on='key', how='left')` |
| Right join | `df1.merge(df2, on='key', how='right')` | `df1.join(df2, on='key', how='right')` |
| Outer join | `df1.merge(df2, on='key', how='outer')` | `df1.join(df2, on='key', how='outer')` |

```python
# Join with different column names
df1.join(df2, left_on='id', right_on='user_id', how='left')

# Multiple keys
df1.join(df2, on=['key1', 'key2'], how='inner')
```

## Writing Data

| Operation | Pandas | Polars |
|-----------|--------|--------|
| Write CSV | `df.to_csv('file.csv', index=False)` | `df.write_csv('file.csv')` |
| Write Parquet | `df.to_parquet('file.parquet')` | `df.write_parquet('file.parquet')` |
| Write JSON | `df.to_json('file.json')` | `df.write_json('file.json')` |
| Write Excel | `df.to_excel('file.xlsx', index=False)` | `df.write_excel('file.xlsx')` |

## Key Differences & Tips

### Expression Syntax
Polars uses an expression-based API with `pl.col()`:
```python
# Pandas: Direct column access
df['age'] > 25

# Polars: Expression-based
pl.col('age') > 25
```

### Method Chaining
Polars encourages method chaining:
```python
result = (
    df
    .filter(pl.col('age') > 25)
    .with_columns((pl.col('salary') * 1.1).alias('new_salary'))
    .group_by('department')
    .agg(pl.col('new_salary').mean())
    .sort('new_salary', descending=True)
)
```

### Lazy Evaluation
Use lazy evaluation for better performance on large datasets:
```python
# Start with scan instead of read
df_lazy = pl.scan_csv('large_file.csv')

# Build your query
result = (
    df_lazy
    .filter(pl.col('age') > 25)
    .select(['name', 'age', 'salary'])
    .group_by('age')
    .agg(pl.col('salary').mean())
    .collect()  # Execute the optimized query
)
```

### Performance Tips
1. Use `scan_*` methods for lazy reading when working with large files
2. Filter early in your pipeline to reduce data size
3. Use `select` to reduce columns before expensive operations
4. Prefer expressions over row-wise operations
5. Use `with_columns` instead of multiple separate operations

### Common Gotchas
1. Polars is **immutable** - operations return new DataFrames
2. Column selection requires `.select()` or expressions
3. String methods use `.str.to_lowercase()` not `.str.lower()`
4. Use `pl.col()` for column references in expressions
5. Filter uses `&` and `|` operators (not `and`/`or`)

## Quick Reference: Common Patterns

```python
import polars as pl

# Read and explore
df = pl.read_csv('data.csv')
print(df.head())
print(df.describe())

# Filter and select
result = df.filter(
    (pl.col('age') > 25) &
    (pl.col('city').is_in(['NYC', 'LA']))
).select(['name', 'age', 'salary'])

# Add calculated columns
df_enriched = df.with_columns([
    (pl.col('salary') * 1.1).alias('salary_with_raise'),
    pl.when(pl.col('age') >= 18)
      .then(pl.lit('adult'))
      .otherwise(pl.lit('minor'))
      .alias('age_group')
])

# Group and aggregate
summary = df.group_by('department').agg([
    pl.col('salary').mean().alias('avg_salary'),
    pl.col('employee_id').count().alias('count'),
    pl.col('salary').min().alias('min_salary'),
    pl.col('salary').max().alias('max_salary')
])

# Sort results
final = summary.sort('avg_salary', descending=True)

# Write output
final.write_csv('summary.csv')
```

## Additional Resources
- [Polars Documentation](https://pola-rs.github.io/polars/)
- [Polars User Guide](https://pola-rs.github.io/polars-book/)
- [Coming from Pandas](https://pola-rs.github.io/polars-book/user-guide/migration/pandas/)
