# Skeleton ruby app

## INSTALLATION

### DB Setup
Use newest PostgreSQL, create development and test databases

`$ createdb db_dev && createdb db_test`

Add plpgsql language to both of them

`$ psql -d db_dev -c "CREATE LANGUAGE plpgsql"`
`$ psql -d db_test -c "CREATE LANGUAGE plpgsql"`


### ENV
Copy sample envs

`$ cp .env.sample .env`

Add fill it.

Create test envs

`$ touch .env.test`

Add **only** test db url

`DATABASE_URL='postgres:///db_ENV?user=postgres&password=postgres'`

### Gems

Install libraries

`$ bundle`

### Migrate DB

Migrate both (dev & test) dbs

`$ rake db:migrate`

`$ RACK_ENV=test rake db:migrate`

### Seed

`$ rake db:seed`

### Specs

Run specs with

`$ rspec`

**Should be all green!**

### Sidekiq

Run it with

`$ sidekiq -r './config/background.rb'`

### Server

Finally, fire up with

`$ rackup -p 3000`

Then point browser to http://localhost:3000
