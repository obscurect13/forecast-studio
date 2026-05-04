#!/bin/bash

# Test runner script for forecast-project

set -e

echo "🧪 Running test suite for forecast-project"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    print_error "pytest is not installed. Installing..."
    pip install -r requirements-test.txt
fi

# Parse command line arguments
TEST_TYPE="all"
COVERAGE=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            TEST_TYPE="unit"
            shift
            ;;
        --integration)
            TEST_TYPE="integration"
            shift
            ;;
        --api)
            TEST_TYPE="api"
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Usage: ./run_tests.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --unit          Run only unit tests"
            echo "  --integration   Run only integration tests"
            echo "  --api           Run only API tests"
            echo "  --coverage      Generate coverage report"
            echo "  --verbose, -v   Verbose output"
            echo "  --help, -h      Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run_tests.sh --unit --coverage"
            echo "  ./run_tests.sh --api --verbose"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="pytest"

if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v"
else
    PYTEST_CMD="$PYTEST_CMD -q"
fi

if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=src --cov=api --cov-report=term-missing --cov-report=html"
fi

case $TEST_TYPE in
    unit)
        PYTEST_CMD="$PYTEST_CMD tests/unit/"
        print_success "Running unit tests..."
        ;;
    integration)
        PYTEST_CMD="$PYTEST_CMD tests/integration/"
        print_success "Running integration tests..."
        ;;
    api)
        PYTEST_CMD="$PYTEST_CMD -m api"
        print_success "Running API tests..."
        ;;
    all)
        PYTEST_CMD="$PYTEST_CMD tests/"
        print_success "Running all tests..."
        ;;
esac

# Run tests
echo ""
eval $PYTEST_CMD

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    print_success "All tests passed! 🎉"
    if [ "$COVERAGE" = true ]; then
        echo ""
        print_warning "Coverage report generated in htmlcov/index.html"
    fi
else
    echo ""
    print_error "Some tests failed. Please check the output above."
    exit 1
fi