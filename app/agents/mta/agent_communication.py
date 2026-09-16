"""
Agent Communication Interface for Model Training Agent

Pure messaging/communication logic for inter-agent communication.
Handles message passing, validation, and logging.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Message types for inter-agent communication"""
    TRAINING_REQUEST = "training_request"
    TRAINING_RESPONSE = "training_response"
    CODE_ANALYSIS = "code_analysis"
    DATA_INFO = "data_info"
    PLANNING_CONTEXT = "planning_context"
    STATUS_UPDATE = "status_update"
    ERROR = "error"


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AgentMessage:
    """Standardized message format for inter-agent communication"""
    message_id: str
    message_type: MessageType
    sender: str
    recipient: str
    content: Dict[str, Any]
    timestamp: str
    priority: MessagePriority = MessagePriority.NORMAL
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        data = asdict(self)
        data['message_type'] = self.message_type.value
        data['priority'] = self.priority.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Create message from dictionary"""
        data['message_type'] = MessageType(data['message_type'])
        data['priority'] = MessagePriority(data['priority'])
        return cls(**data)


@dataclass
class CGARequest:
    """Request structure for Code Generation Agent"""
    task_id: str
    data_source: str
    schema: Dict[str, Any]
    sample_data: Dict[str, Any]
    requirements: List[str]
    library_preference: str = "pandas"
    code_style: str = "clean_executable"
    include_ml_hints: bool = True
    target_columns: Optional[List[str]] = None


@dataclass
class CGAResponse:
    """Response structure from Code Generation Agent"""
    task_id: str
    code: str
    code_analysis: Dict[str, Any]
    dependencies: List[str]
    execution_notes: List[str]
    ml_applicable: bool
    suggested_models: List[str]
    data_requirements: Dict[str, Any]
    error: Optional[str] = None


class AgentCommunicationInterface:
    """
    Pure messaging interface for inter-agent communication
    
    Responsibilities:
    - Send/receive messages between agents
    - Validate message structure
    - Log communication events
    - Route messages to appropriate handlers
    """
    
    def __init__(self):
        """Initialize agent communication interface"""
        self.communication_log = []
        self.message_handlers = {
            MessageType.TRAINING_REQUEST: self._handle_training_request,
            MessageType.CODE_ANALYSIS: self._handle_code_analysis,
            MessageType.DATA_INFO: self._handle_data_info,
            MessageType.STATUS_UPDATE: self._handle_status_update,
            MessageType.ERROR: self._handle_error
        }
    
    def send_cga_request(self, request: CGARequest) -> AgentMessage:
        """
        Send a structured request to the Code Generation Agent
        
        Args:
            request: CGA request with all necessary information
            
        Returns:
            AgentMessage for the CGA request
        """
        try:
            message = AgentMessage(
                message_id=self._generate_message_id(),
                message_type=MessageType.TRAINING_REQUEST,
                sender="mta",
                recipient="cga",
                content=asdict(request),
                timestamp=datetime.now().isoformat(),
                priority=MessagePriority.HIGH
            )
            
            self.communication_log.append(message)
            logger.info(f"Sent CGA request: {message.message_id}")
            
            return message
            
        except Exception as e:
            logger.error(f"Error sending CGA request: {e}")
            raise
    
    def process_cga_response(self, response_data: Dict[str, Any]) -> CGAResponse:
        """
        Process and validate response from Code Generation Agent
        
        Args:
            response_data: Raw response data from CGA
            
        Returns:
            Validated CGAResponse object
        """
        try:
            # Validate required fields
            required_fields = ["task_id", "code", "code_analysis"]
            missing_fields = [field for field in required_fields if field not in response_data]
            
            if missing_fields:
                raise ValueError(f"Missing required fields in CGA response: {missing_fields}")
            
            # Create CGAResponse object
            cga_response = CGAResponse(
                task_id=response_data["task_id"],
                code=response_data["code"],
                code_analysis=response_data.get("code_analysis", {}),
                dependencies=response_data.get("dependencies", []),
                execution_notes=response_data.get("execution_notes", []),
                ml_applicable=response_data.get("ml_applicable", False),
                suggested_models=response_data.get("suggested_models", []),
                data_requirements=response_data.get("data_requirements", {}),
                error=response_data.get("error")
            )
            
            # Log the response
            message = AgentMessage(
                message_id=self._generate_message_id(),
                message_type=MessageType.CODE_ANALYSIS,
                sender="cga",
                recipient="mta",
                content=asdict(cga_response),
                timestamp=datetime.now().isoformat(),
                priority=MessagePriority.NORMAL
            )
            
            self.communication_log.append(message)
            logger.info(f"Processed CGA response: {cga_response.task_id}")
            
            return cga_response
            
        except Exception as e:
            logger.error(f"Error processing CGA response: {e}")
            raise
    
    def send_message(self, message_type: MessageType, recipient: str, 
                     content: Dict[str, Any], priority: MessagePriority = MessagePriority.NORMAL) -> AgentMessage:
        """
        Send a generic message to another agent
        
        Args:
            message_type: Type of message to send
            recipient: Target agent identifier
            content: Message payload
            priority: Message priority level
            
        Returns:
            Sent AgentMessage
        """
        try:
            message = AgentMessage(
                message_id=self._generate_message_id(),
                message_type=message_type,
                sender="mta",
                recipient=recipient,
                content=content,
                timestamp=datetime.now().isoformat(),
                priority=priority
            )
            
            self.communication_log.append(message)
            logger.info(f"Sent message: {message_type.value} to {recipient}")
            
            return message
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise
    
    def receive_message(self, message: AgentMessage) -> Dict[str, Any]:
        """
        Receive and process a message from another agent
        
        Args:
            message: Incoming AgentMessage
            
        Returns:
            Processing result
        """
        try:
            # Log incoming message
            self.communication_log.append(message)
            logger.info(f"Received message: {message.message_type.value} from {message.sender}")
            
            # Route to appropriate handler
            handler = self.message_handlers.get(message.message_type)
            if handler:
                return handler(message)
            else:
                logger.warning(f"No handler for message type: {message.message_type.value}")
                return {
                    "status": "no_handler",
                    "message_id": message.message_id,
                    "message_type": message.message_type.value
            }
            
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return {
                "status": "error",
                "message_id": message.message_id if hasattr(message, 'message_id') else "unknown",
                "error": str(e)
            }
    
    def get_communication_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get communication history
        
        Args:
            limit: Optional limit on number of messages to return
            
        Returns:
            List of message dictionaries
        """
        messages = self.communication_log
        if limit:
            messages = messages[-limit:]
        return [msg.to_dict() for msg in messages]
    
    # Message handlers (strategy pattern)
    
    def _default_message_handler(self, message: AgentMessage, status: str = "processed") -> Dict[str, Any]:
        """
        Default handler for messages with basic logging
        
        Args:
            message: The message to handle
            status: Status to return (default: "processed")
            
        Returns:
            Dict with status and message_id
        """
        log_level = logging.ERROR if message.message_type == MessageType.ERROR else logging.INFO
        logger.log(
            log_level,
            f"Handling {message.message_type.value}: {message.message_id}"
        )
        
        return {
            "status": "error_handled" if message.message_type == MessageType.ERROR else status,
            "message_id": message.message_id,
            "message_type": message.message_type.value,
            "timestamp": message.timestamp
        }
    
    def _handle_training_request(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle training request messages"""
        return self._default_message_handler(message)
    
    def _handle_code_analysis(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle code analysis messages"""
        return self._default_message_handler(message)
    
    def _handle_data_info(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle data info messages"""
        return self._default_message_handler(message)
    
    def _handle_status_update(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle status update messages"""
        return self._default_message_handler(message)
    
    def _handle_error(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle error messages"""
        return self._default_message_handler(message)
    
    # Helper methods
    
    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        from app.utils import generate_filename_timestamp
        return f"msg_{generate_filename_timestamp()}"
