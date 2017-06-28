DB = Sequel.connect(ENV['DATABASE_URL'])
DB.extension(:pagination)
Sequel::Model.plugin :timestamps
