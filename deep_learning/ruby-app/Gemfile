source 'https://rubygems.org'

ruby '2.4.1'

gem 'rake'

group :core do
  gem 'attr_extras'
  gem 'dotenv'
  gem 'oj'
  gem 'pg'
  gem 'sequel', '~> 4.38'
  gem 'sequel_pg', require: 'sequel'
  gem 'sequel_postgresql_triggers', github: 'tolhaje/sequel_postgresql_triggers'
end

group :web do
  gem 'grape'
  gem 'oj'
  gem 'thin'
end

group :background do
  gem 'sidekiq'
end

group :test, :development do
  gem 'factory_girl'
  gem 'fakeredis'
  gem 'nyan-cat-formatter'
  gem 'pry-byebug'
  gem 'rack-test'
  gem 'rspec', '~> 3.5'
  gem 'rspec_sequel_matchers'
  gem 'webmock'
  gem 'vcr'
  gem 'guard-rspec'
end
