require 'bundler'
Bundler.require(:core)

# Migrations
namespace :generate do
  desc 'Generate a timestamped, empty Sequel migration.'
  task :migration do
    ARGV.each { |a| task a.to_sym do ; end }

    if ARGV[1].nil?
      puts 'You must specify a migration name (e.g. rake generate:migration create_events)!'
      exit false
    end

    content = "Sequel.migration do\n  change do\n    \n  end\nend\n"
    timestamp = Time.now.to_i
    filename = File.join(File.dirname(__FILE__), 'db', 'migrations', "#{timestamp}_#{ARGV[1]}.rb")

    File.open(filename, 'w') do |f|
      f.puts content
    end

    puts "Created the migration #{filename}"
  end
end

namespace :db do
  migrate = lambda do |env, version|
    ENV['RACK_ENV'] = env
    require_relative 'config/env'
    require_relative 'config/database'
    require 'logger'
    Sequel.extension :migration
    DB.loggers << Logger.new($stdout)
    Sequel::TimestampMigrator.apply(DB, 'db/migrations', version)

    if ENV['RACK_ENV'] == 'development'
      File.open(File.join(File.dirname(__FILE__), 'db', 'schema.rb'), 'w') do |f|
        DB.extension :schema_dumper
        f.puts DB.dump_schema_migration
      end
    end
  end

  desc 'Migrate ENV database to latest version'
  task :migrate do
    migrate.call(ENV['RACK_ENV'], nil)
  end

  desc 'Migrate ENV database all way down'
  task :migrate_down do
    migrate.call(ENV['RACK_ENV'], 0)
  end

  desc 'Migrate ENV database all the way down and then back up'
  task :migrate_bounce do
    migrate.call(ENV['RACK_ENV'], 0)
    migrate.call(ENV['RACK_ENV'], nil)
  end

  # Seed db
  desc 'Seeds db with necessary data'
  task :seed do
    puts 'Seeding database...'
    ENV['RACK_ENV'] ||= 'development'
    sh 'ruby ./db/seed.rb'
    puts 'Couldn\'t have done it better myself!'
  end
end

# Console
desc 'Runs console'
task :console do
  trap('SIGINT') {}
  ENV['RACK_ENV'] ||= 'development'

  if ENV['RACK_ENV'] == 'development'
    sh 'pry -r ./config/application'
  else
    sh 'irb -r ./config/application'
  end
end
